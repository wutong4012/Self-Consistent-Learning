from curses import raw
import gc

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
    dis_tokenizer = BertTokenizer.from_pretrained(config.discriminator)
    discriminator = Discriminator(config).cuda().eval()
    
    predict_dataset = SimGanDataset(raw_dataset)
    def collate_fn(batch_data):
        return dis_pred_collate(batch_data, dis_tokenizer)
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=128,
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

        threshold0 = 0.9 - (config.cycle + 1) * 0.04
        if threshold0 < config.gen_threshold0:
            threshold0 = config.gen_threshold0
        threshold1 = 0.9 - (config.cycle + 1) * 0.04
        if threshold1 < config.gen_threshold1:
            threshold1 = config.gen_threshold1
        all_logits = torch.cat(all_logits, dim=0)
        assert all_logits.size(0) == raw_dataset.num_rows
        
        for idx in range(raw_dataset.num_rows):
            if all_logits[idx][0] >= threshold0:
                scores.append(0)
                text1.append(raw_dataset['text1'][idx])
                text2.append(raw_dataset['text2'][idx])
            elif all_logits[idx][1] >= threshold1:
                scores.append(1)
                text1.append(raw_dataset['text1'][idx])
                text2.append(raw_dataset['text2'][idx])

    discriminator.to('cpu')
    if rank == 0:
        print(f'**********Origin Generate Data Samples is {raw_dataset.num_rows}**********')
        print(f'**********The Threshold 0 is {threshold0}**********')
        print(f'**********There are {scores.count(0)} Samples to be Selected 0!**********')
        print(f'**********The Threshold 1 is {threshold1}**********')
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

    raw_text, sim_text, real_ppl_list = [], [], []
    for idx, item in enumerate(sim_sentence):
        if item.count('”的相似句是“') != 1 or (
                item.count('“') % 2 != 0 or item.count('”') % 2 != 0):
            continue

        item = item.replace(' ', '').split('”的相似句是“')
        raw_text.append(item[0][1:])
        sim_text.append(item[1][:-1])
    #     real_ppl_list.append(-output_dict['ppl_list'][idx])  # 加上负号，使其越大越好

    # ppl_list = torch.softmax(
    #     torch.tensor(real_ppl_list), dim=0).numpy().tolist()
    raw_dataset = Dataset.from_dict({
        'text1': raw_text,
        'text2': sim_text,
    })
    
    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the spliting')
        torch.distributed.barrier()
    raw_dataset = raw_dataset.train_test_split(
        test_size=0.5, seed=config.seed,
        train_indices_cache_file_name=config.cache_data_path+'/split_cache_'+str(config.cycle))
    score_ds, train_ds = raw_dataset['train'], raw_dataset['test']
    if rank == 0:
        new_data_path = config.sim_data_path + f'/score_cycle_{config.cycle + 1}'
        print(f'**********Generate Data For Score Samples is {score_ds.num_rows}**********')
        score_ds.save_to_disk(new_data_path)
        torch.distributed.barrier()

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
    
    threshold = 0.9 - (config.cycle + 1) * 0.04
    if threshold < config.dis_threshold:
        threshold = config.dis_threshold
    if rank == 0:
        print(f'**********Start to Post Process the Scored Data, \
            threshold is {threshold}**********')

    text1, text2, scores = [], [], []
    for idx, item in enumerate(dis_output_dict['logits']):
        if item[1] >= threshold:
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
        cur_input_ids = gen_tokenizer(
            '<bos>“' + item['sentence'] + '”的相似句是“', return_tensors='pt'
        ).input_ids.squeeze()[:-1]  # 不能加<eos>

        # 每个样本复制 N 份
        length = [cur_input_ids.size(0)] * config.gen_repeat_times
        cur_input_ids = [cur_input_ids] * config.gen_repeat_times

        length_list.extend(length)
        input_ids.extend(cur_input_ids)

    input_ids = pad_sequence(
        [x for x in input_ids], batch_first=True, padding_value=50000)
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


def create_predict_dataloader(config, tokenizer, rank, attri):
    if attri == 'gen':
        batch_size = config.pre_gen_bs

        test_ds = datasets.load_from_disk(config.test_sentence_path + config.data_name + '_sentence')
        if config.data_name == 'qqp':
            start = config.data_num * 3000 % 9000
            end = (config.data_num + 1) * 3000 % 9000
        
        else:
            integer = int(test_ds.num_rows / 10000) * 10000
            start = config.data_num * 20000 % integer
            end = (config.data_num + 1) * 20000 % integer

        if end == 0:
            end = test_ds.num_rows
        sentence_ds = test_ds.select(range(start, end))
        if rank == 0:
            print(f'**********The Test_ds {integer} Range is {start} ~~ {end}**********')

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
