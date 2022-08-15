import os
import gc
import requests
from sklearn.model_selection import train_test_split

import torch
import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

from data_utlis.sim_gen_dataset import load_data, SimGanDataset
from model_utils.sim_gen_model import Discriminator


feats = datasets.Features({"text1": datasets.Value('string'),
                           "text2": datasets.Value('string'),
                           "score": datasets.Value('int8')})


def multiply_pre_score(config, raw_dataset, rank):
    dis_tokenizer = BertTokenizer.from_pretrained(
        config.dis_model_path + config.discriminator)
    discriminator = Discriminator(config).cuda().eval()
    
    predict_dataset = SimGanDataset(raw_dataset)
    def collate_fn(batch_data):
        return dis_pred_collate(batch_data, dis_tokenizer)
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=384,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    with torch.no_grad():
        all_logits, text1, text2, scores = [], [], [], []
        for batch in dataloader:
            torch.cuda.empty_cache()
            logits = discriminator.forward(
                batch['input_ids'].cuda(), None)
            all_logits.append(torch.softmax(logits, dim=1))

        threshold0 = config.min_thre0 + config.cycle * 0.07
        if threshold0 > config.max_thre0:
            threshold0 = config.max_thre0
        threshold1 = config.min_thre1 + config.cycle * 0.07
        if threshold1 > config.max_thre1:
            threshold1 = config.max_thre1

        all_logits = torch.cat(all_logits, dim=0)
        assert all_logits.size(0) == raw_dataset.num_rows
        
        for idx in range(raw_dataset.num_rows):
            scores.append(1)
            text1.append(raw_dataset['text1'][idx])
            text2.append(raw_dataset['text2'][idx])
            
            # if all_logits[idx][0] >= threshold0:
            #     scores.append(0)
            #     text1.append(raw_dataset['text1'][idx])
            #     text2.append(raw_dataset['text2'][idx])
            # elif all_logits[idx][1] >= threshold1:
            #     scores.append(1)
            #     text1.append(raw_dataset['text1'][idx])
            #     text2.append(raw_dataset['text2'][idx])

    discriminator.to('cpu')
    if rank == 0:
        print(f'**********Origin Generate Data Samples is {raw_dataset.num_rows}**********')
        print(f'**********The Threshold 0 is {threshold0}**********')
        print(f'**********The Threshold 1 is {threshold1}**********')
        print(f'**********There are {scores.count(0)} Samples to be Selected 0!**********')
        print(f'**********There are {scores.count(1)} Samples to be Selected 1!**********')

    return {
        'text1': text1,
        'text2': text2,
        'score': scores,
    }


def gen_postprocess(output_dict, gen_tokenizer, config, rank):
    gc.collect()
    torch.cuda.empty_cache()
    
    if rank == 0:
        print('**********Start to Post Process the Generated Data**********')
    sim_sentence = gen_tokenizer.batch_decode(
        output_dict['ids_list'], skip_special_tokens=True)

    if config.chinese:
        raw_text, sim_text, real_ppl_list = [], [], []
        for idx, item in enumerate(sim_sentence):
            if item.count('”的相似句是“') != 1 or (
                    item.count('“') % 2 != 0 or item.count('”') % 2 != 0):
                continue

            item = item.replace(' ', '').split('”的相似句是“')
            raw_text.append(item[0][1:])
            sim_text.append(item[1][:-1])
    
    else:
        raw_text, sim_text = [], []
        for item in sim_sentence:
            item = item.split('\" is similar to \"')
            if len(item) != 2 or item[0][1:] == item[1][:-1]:
                continue
            
            raw_text.append(item[0][1:])
            sim_text.append(item[1][:-1])

    raw_dataset = Dataset.from_dict({
        'text1': raw_text,
        'text2': sim_text,
    })
    
    # dataset自带的train_test_split多卡时有bug
    # score_dict, train_dict = train_test_split(
    #     raw_dataset, test_size=0.5, random_state=config.seed+config.cycle)
    # score_ds, train_ds = Dataset.from_dict(score_dict), Dataset.from_dict(train_dict)
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
    
    dis_threshold = config.min_dis_thre + (config.cycle + 1) * 0.07
    if dis_threshold > config.max_dis_thre:
        dis_threshold = config.max_dis_thre
    if rank == 0:
        print(f'**********Start to Post Process the Scored Data, \
            threshold is {dis_threshold}**********')

    text1, text2, scores = [], [], []
    for idx, item in enumerate(dis_output_dict['logits']):
        if item[1] >= dis_threshold:
            scores.append(1)
            text1.append(dis_output_dict['text1'][idx])
            text2.append(dis_output_dict['text2'][idx])

        else:
            scores.append(0)
            text1.append(dis_output_dict['text1'][idx])
            text2.append(dis_output_dict['text2'][idx])

    dis_dataset = Dataset.from_dict({
        'text1': text1,
        'text2': text2,
        'score': scores,
    }, features=feats)

    if rank == 0:
        new_data_path = config.sim_data_path + f'/trainG_cycle_{config.cycle + 1}'
        print(f'**********Score Data 1 Samples is {scores.count(1)}**********')
        dis_dataset.save_to_disk(new_data_path)
        print('**********Score Data: done!!!**********')


def gen_pred_collate(batch_data, gen_tokenizer, config):
    input_ids, length_list = [], []
    for item in batch_data:
        if config.chinese:
            cur_input_ids = gen_tokenizer(
                '<bos>“' + item['sentence'] + '”的相似句是“', return_tensors='pt'
            ).input_ids.squeeze()[:-1]  # 不能加<eos>
        else:
            cur_input_ids = gen_tokenizer(
                '"' + item['sentence'] + '" is similar to "', return_tensors='pt'
            ).input_ids.squeeze()[1:]  # 去掉<bos>

        # 每个样本复制 N 份
        length = [cur_input_ids.size(0)] * config.gen_repeat_times
        cur_input_ids = [cur_input_ids] * config.gen_repeat_times

        length_list.extend(length)
        input_ids.extend(cur_input_ids)

    if config.chinese:
        input_ids = pad_sequence(
            [x for x in input_ids], batch_first=True, padding_value=50000)
    else:
        input_ids = pad_sequence(
            [x for x in input_ids], batch_first=True, padding_value=1)
    length_tensor = torch.tensor(length_list)

    return {
        'input_ids': input_ids,
        'length_tensor': length_tensor,
    }


def dis_pred_collate(batch_data, dis_tokenizer):
    dis_input_ids, text1, text2 = [], [], []
    for item in batch_data:
        text1.append(item['text1'])
        text2.append(item['text2'])

        dis_text = item['text1'] + '[SEP]' + item['text2']
        input_ids = dis_tokenizer(
            dis_text, return_tensors='pt').input_ids.squeeze()
        dis_input_ids.append(input_ids)

    dis_input_ids = pad_sequence([x for x in dis_input_ids],
                                 batch_first=True, padding_value=0)

    return {
        'text1': text1,
        'text2': text2,
        'input_ids': dis_input_ids,
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
    if gen_ds.num_rows > 6000:
        gen_ds = gen_ds.select(range(6000))
    
    def process_gen(example):
        sentence_list = []
        for item in example['text2']:
            sentence_list.append(item)
        return {'sentence': sentence_list}
    
    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()

    gen_ds = gen_ds.map(process_gen, batch_size=500, batched=True,
        cache_file_name=config.cache_data_path+'/gen_cache'+str(config.cycle))

    if rank == 0:
        torch.distributed.barrier()
    
    return gen_ds


def create_predict_dataloader(config, tokenizer, rank, attri):
    if attri == 'gen':
        batch_size = config.pre_gen_bs

        test_ds = datasets.load_from_disk(config.test_sentence_path + config.data_name + '_sentence')
        config.start = config.end
        if config.start == test_ds.num_rows:
            config.start = config.end = 0
        config.end += config.sentence_num
        if config.end > test_ds.num_rows:
            config.end = test_ds.num_rows
            
        origin_ds = test_ds.select(range(config.start, config.end))
        if config.vae2gen:
            vae_path = config.test_sentence_path + 'vae_sentence/' + str(config.idx) + '/sent_' + str(config.cycle)
            if rank == 0 and (not os.path.exists(vae_path)):
                print('**********Starting use VAE server...**********')
                print(f'**********std_scale is {config.std_scale}**********')
                get_vae_sent(config, origin_ds, vae_path)
                print('**********Generate Senteces Finished!**********')
            torch.distributed.barrier()
            extra_ds = datasets.load_from_disk(vae_path)
        elif config.txl2gen:
            if config.cycle == -1:
                sentence_ds = test_ds.select(range(0, config.sentence_num*2))
                config.end = config.sentence_num * 2
            else:
                extra_ds = process_gen_ds(config, rank)
        if (config.txl2gen and config.cycle != -1) or config.vae2gen:
            sentence_ds = datasets.concatenate_datasets([origin_ds, extra_ds])
        if rank == 0:
            print(f'**********The Test_ds Range is {config.start} ~~ {config.end}**********')

        predict_dataset = SimGanDataset(sentence_ds)

        def collate_fn(batch_data):
            return gen_pred_collate(batch_data, tokenizer, config)

    elif attri == 'dis':
        batch_size = config.pre_dis_bs

        gen_ds = load_data(config, rank, is_score=True, attri='dis')
        predict_dataset = SimGanDataset(gen_ds)

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
