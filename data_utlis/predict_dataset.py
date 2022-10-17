import gc
import random
import requests

import torch
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AlbertTokenizer

from data_utlis.sim_data_collate import padding_dis_mask
from data_utlis.sim_gen_dataset import load_data, preprocess, SimGanDataset
from model_utils.sim_gen_model import Discriminator


feats = datasets.Features({"sentence1": datasets.Value('string'),
                           "sentence2": datasets.Value('string'),
                           "label": datasets.Value('int8')})


def multiply_pre_score(config, raw_dataset, rank):
    dis_tokenizer = AutoTokenizer.from_pretrained(
        '/cognitive_comp/wutong/source/model_base/pretrained_zh/' + config.bustm_model)
    discriminator = Discriminator(config, dis_tokenizer).cuda().eval()
    
    torch.distributed.barrier()
    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()
    raw_dataset = raw_dataset.map(preprocess, 
                                  cache_file_name=config.cache_data_path+'/raw_dataset_cache'+str(config.cycle))
    if rank == 0:
        torch.distributed.barrier()
    
    predict_dataset = SimGanDataset(data=raw_dataset, tokenizer=dis_tokenizer, predict=True)
    def collate_fn(batch_data):
            return dis_pred_collate(batch_data, dis_tokenizer)
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=config.pre_dis_bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    with torch.no_grad():
        raw_text1, raw_text2 = [], []
        all_prob, text1, text2, scores = [], [], [], []
        for batch in dataloader:
            torch.cuda.empty_cache()
            prob = discriminator.forward(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                token_type_ids=batch['token_type_ids'].cuda(),
                position_ids=batch['position_ids'].cuda(),
                clslabels_mask=batch['clslabels_mask'].cuda(),
                # bt_label_idx=batch['label_idx'].cuda()
            )
            for item in prob:
                all_prob.append(torch.tensor([item[1], 1-item[1]]))
            raw_text1.extend(batch['sentence1'])
            raw_text2.extend(batch['sentence2'])

        threshold0 = config.min_thre0 + config.cycle * config.add_thre
        if threshold0 > config.max_thre0:
            threshold0 = config.max_thre0
        threshold1 = config.min_thre1 + config.cycle * config.add_thre
        if threshold1 > config.max_thre1:
            threshold1 = config.max_thre1

        all_prob = torch.stack(all_prob)
        assert all_prob.size(0) == len(raw_text1)
        
        for idx in range(len(raw_text1)):
            if config.consistency:
                scores.append(1)
                text1.append(raw_dataset['sentence1'][idx])
                text2.append(raw_dataset['sentence2'][idx])
            
            else:
                if all_prob[idx][0] >= threshold0:
                    scores.append(0)
                    text1.append(raw_text1[idx])
                    text2.append(raw_text2[idx])
                elif all_prob[idx][1] >= threshold1:
                    scores.append(1)
                    text1.append(raw_text1[idx])
                    text2.append(raw_text2[idx])

    discriminator.to('cpu')
    if rank == 0:
        print(f'**********Origin Generate Data Samples is {len(raw_text1)}**********')
        print(f'**********The Threshold 0 is {threshold0}**********')
        print(f'**********The Threshold 1 is {threshold1}**********')
        print(f'**********There are {scores.count(0)} Samples to be Selected 0!**********')
        print(f'**********There are {scores.count(1)} Samples to be Selected 1!**********')

    return {
        'sentence1': text1,
        'sentence2': text2,
        'label': scores,
    }


def gen_postprocess(output_dict, gen_tokenizer, config, rank):
    gc.collect()
    torch.cuda.empty_cache()
    
    if rank == 0:
        print('**********Start to Post Process the Generated Data**********')
    sim_sentence = gen_tokenizer.batch_decode(
        output_dict['ids_list'], skip_special_tokens=True)

    raw_text, sim_text = [], []
    for item in sim_sentence:
        if item.count('”的相似句是“') != 1 or (
                item.count('“') % 2 != 0 or item.count('”') % 2 != 0):
            continue

        item = item.replace(' ', '').split('”的相似句是“')
        if (len(item[0][1:]) + len(item[1][:-1]) * 2) > 400:
            continue 
        raw_text.append(item[0][1:])
        sim_text.append(item[1][:-1])

    raw_dataset = Dataset.from_dict({
        'sentence1': raw_text,
        'sentence2': sim_text,
    })
    
    score_ds, train_ds = raw_dataset, raw_dataset
    if rank == 0:
        new_data_path = config.sim_data_path + f'/score_cycle_{config.cycle + 1}'
        print(f'**********Generate Data For Score Samples is {score_ds.num_rows}**********')
        score_ds.save_to_disk(new_data_path)

    if config.cycle >= 0:
        gen_ds_dict = multiply_pre_score(config, train_ds, rank)
        gen_dataset = Dataset.from_dict(gen_ds_dict, features=feats)

        if rank == 0:
            new_data_path = config.sim_data_path + f'/trainD_cycle_{config.cycle}'
            print(f'**********Generate Data For TrainD Samples is {gen_dataset.num_rows}**********')
            gen_dataset.save_to_disk(new_data_path)
            print('**********Gen Data: done!!!**********')


def dis_postprocess(dis_output_dict, config, rank):
    gc.collect()
    torch.cuda.empty_cache()
    
    dis_threshold = config.min_dis_thre + (config.cycle + 1) * config.add_thre
    if dis_threshold > config.max_dis_thre:
        dis_threshold = config.max_dis_thre
    if rank == 0:
        print(f'**********Start to Post Process the Scored Data, \
            threshold is {dis_threshold}**********')

    text1, text2, scores = [], [], []
    for idx, item in enumerate(dis_output_dict['prob']):
        if 1 - item[1] >= dis_threshold:
            scores.append(1)
            text1.append(dis_output_dict['sentence1'][idx])
            text2.append(dis_output_dict['sentence2'][idx])

        else:
            scores.append(0)
            text1.append(dis_output_dict['sentence1'][idx])
            text2.append(dis_output_dict['sentence2'][idx])

    dis_dataset = Dataset.from_dict({
        'sentence1': text1,
        'sentence2': text2,
        'label': scores,
    }, features=feats)

    if rank == 0:
        new_data_path = config.sim_data_path + f'/trainG_cycle_{config.cycle + 1}'
        print(f'**********Score Data 1 Samples is {scores.count(1)}**********')
        dis_dataset.save_to_disk(new_data_path)
        print('**********Score Data: done!!!**********')


def dis_pred_collate(batch_data, tokenizer):
    max_length = 0
    input_ids, token_type_ids, attention_mask, position_ids = [], [], [], []
    clslabels_mask, sentence1, sentence2, label_idx = [], [], [], []
    for item in batch_data:
        max_length = max(max_length, item['attention_mask'].size(0))
        input_ids.append(item['input_ids'])
        token_type_ids.append(item['token_type_ids'])
        attention_mask.append(item['attention_mask'])
        position_ids.append(item['position_ids'])
        clslabels_mask.append(item['clslabels_mask'])
        # label_idx.append(item['label_idx'])
        sentence1.append(item['sentence1'])
        sentence2.append(item['sentence2'])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask = padding_dis_mask(attention_mask, max_length)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
    clslabels_mask = pad_sequence(clslabels_mask, batch_first=True, padding_value=-10000)
        
    return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "clslabels_mask": clslabels_mask,
            # 'label_idx': torch.stack(label_idx),
            'sentence1': sentence1,
            'sentence2': sentence2
        }


def gen_pred_collate(batch_data, gen_tokenizer, config):
    input_ids, length_list = [], []
    for item in batch_data:
        cur_input_ids = gen_tokenizer(
            '<bos>“' + item['sentence'] + '”的相似句是“', return_tensors='pt'
        ).input_ids.squeeze()[:-1]  # 不能加<eos>

        length = [cur_input_ids.size(0)]
        cur_input_ids = [cur_input_ids]

        length_list.extend(length)
        input_ids.extend(cur_input_ids)

    input_ids = pad_sequence(
        [x for x in input_ids], batch_first=True, padding_value=50000)
    length_tensor = torch.tensor(length_list)

    return {
        'input_ids': input_ids,
        'length_tensor': length_tensor,
    }


def get_vae_sent(config, origin_ds, vae_path):
    sents_list = []
    for idx in range(origin_ds.num_rows):
        # if len(origin_ds[idx]['sentence']) >= 10:
        sents_list.append(origin_ds[idx]['sentence'])
    
    url="http://192.168.52.173:23628/davae"
    result = requests.post(url,
                json={
                    'sent_inputs': sents_list,
                    'top_p': 0.95,
                    'std_scale': config.std_scale,
                    'augm_num':1
                }
            ).json()
    
    vae_ds = Dataset.from_dict(
        {'sentence': result['generated_sentence']}
    )
    vae_ds.save_to_disk(vae_path)
    print(f'**********vae sent_num is {vae_ds.num_rows}**********')


def process_gen_ds(config, rank):
    """
        使用Generator生成的句子再作为预测的输入句子
    """
    gen_ds = datasets.load_from_disk(config.sim_data_path + f'/trainG_cycle_{config.cycle}')
    if gen_ds.num_rows > 3000:
        random_list = random.sample(range(gen_ds.num_rows), 3000)
        gen_ds = gen_ds.select(random_list)
    
    def process_gen(example):
        return {'sentence': example['sentence2']}
    
    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()

    gen_ds = gen_ds.map(process_gen,
                        cache_file_name=config.cache_data_path+'/gen_cache'+str(config.cycle))

    if rank == 0:
        torch.distributed.barrier()
    
    return gen_ds


def create_predict_dataloader(config, tokenizer, rank, attri):
    if attri == 'gen':
        batch_size = config.pre_gen_bs

        test_ds = datasets.Dataset.from_json(
            '/cognitive_comp/wutong/source/sim_data/raw_data/bustm/new_unlabel.json')
        config.start = config.end
        if config.start == test_ds.num_rows:
            config.start = config.end = 0
        config.end += config.sentence_num
        if config.end > test_ds.num_rows:
            config.end = test_ds.num_rows
            
        origin_ds = test_ds.select(range(config.start, config.end))
        if config.cycle == -1:
            sentence_ds = test_ds.select(range(0, config.sentence_num*2))
            config.end = config.sentence_num * 2
        else:
            extra_ds = process_gen_ds(config, rank)
        if (config.txl2gen and config.cycle != -1) or config.vae2gen:
            sentence_ds = datasets.concatenate_datasets([origin_ds, extra_ds])
        if rank == 0:
            print(f'**********The Test_ds Range is {config.start} ~~ {config.end}**********')
        
        predict_dataset = SimGanDataset(sentence_ds, tokenizer=tokenizer, 
                                        predict=True, is_gen=True)

        def collate_fn(batch_data):
            return gen_pred_collate(batch_data, tokenizer, config)

    elif attri == 'dis':
        batch_size = config.pre_dis_bs

        gen_ds = load_data(config, rank, is_score=True, attri='dis')
        
        if rank > 0:
            print(f'Rank {rank} waiting for main process to perform the mapping')
            torch.distributed.barrier()
        gen_ds = gen_ds.map(preprocess, 
                            cache_file_name=config.cache_data_path+'/gen_ds_cache'+str(config.cycle))
        if rank == 0:
            torch.distributed.barrier()
        
        predict_dataset = SimGanDataset(gen_ds, tokenizer=tokenizer, predict=True)
        def collate_fn(batch_data):
            return dis_pred_collate(batch_data, tokenizer)

    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader
