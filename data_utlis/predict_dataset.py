import os
import gc
import random

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

        threshold0 = config.gen_threshold0  # - (config.cycle + 1) * 0.04
        if threshold0 < 0.7:
            threshold0 = 0.7
        threshold1 = config.gen_threshold1 - (config.cycle + 1) * 0.04
        if threshold1 < 0.6:
            threshold1 = 0.6
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


def gen_postprocess(output_dict, gen_tokenizer, config, rank, for_score=False):
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
    
    if for_score:
        if rank == 0:
            new_data_path = config.sim_data_path + f'/score_cycle_{config.cycle + 1}'
            print(f'**********Generate Data Samples is {raw_dataset.num_rows}**********')
            raw_dataset.save_to_disk(new_data_path)
            print('**********Gen Data: done!!!**********')
    
    else:
        gen_ds_dict = multiply_pre_score(config, raw_dataset, rank)
        gen_dataset = Dataset.from_dict(gen_ds_dict, features=feats)

        if rank == 0:
            new_data_path = config.sim_data_path + f'/trainD_cycle_{config.cycle + 1}'
            print(f'**********Generate Data Samples is {gen_dataset.num_rows}**********')
            gen_dataset.save_to_disk(new_data_path)
            print('**********Gen Data: done!!!**********')


def dis_postprocess(dis_output_dict, config, rank):
    gc.collect()
    torch.cuda.empty_cache()
    
    if rank == 0:
        print('**********Start to Post Process the Scored Data**********')
    text1, text2, scores = [], [], []
    for idx, item in enumerate(dis_output_dict['logits']):
        if item[1] >= config.dis_threshold:
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
        item = item['sentence_list']
        if item is None or item == []:
            continue

        # wudao_ds 每段话只随机选一条句子
        if len(item) > 1:
            flag = 1
            random_num = random.sample(range(len(item)), 1)[0]
            for idx in range(0, len(item)):
                if len(item[idx]) < 5 or len(item[idx]) > 50:
                    continue
                else:
                    random_num = idx
                    flag = 0
                    break
            if flag:
                continue
        # test_ds 列表里只有一句话
        elif len(item) == 1:
            if config.use_afqmc:
                if len(item[0]) < 5 or len(item[0]) > 50:
                    continue
            random_num = 0

        cur_input_ids = gen_tokenizer(
            '<bos>“' + item[random_num] + '”的相似句是“', return_tensors='pt'
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
        
        if config.only_use_wudao:
            sentence_ds = load_data(config, rank, is_wudao=True)

        else:
            if config.use_bustm:
                wudao_ds = load_data(config, rank, is_wudao=True)
                test_ds = datasets.load_from_disk(config.test_sentence_path + '/bustm_sentence')
                start = config.data_num * config.bustm_sentence_num % 8000
                end = (config.data_num + 1) * config.bustm_sentence_num % 8000
            
            elif config.use_afqmc:
                test_ds = datasets.load_from_disk(config.test_sentence_path + '/afqmc_sentence')
                start = config.data_num * config.afqmc_sentence_num % 65000
                end = (config.data_num + 1) * config.afqmc_sentence_num % 65000

            if rank > 0:
                torch.distributed.barrier()
            test_ds = test_ds.shuffle(
                config.seed + config.data_num, 
                indices_cache_file_name=config.cache_data_path+'/test_shuffle_'+str(config.data_num))
            if rank == 0:
                torch.distributed.barrier()
            if end == 0:
                end = test_ds.num_rows

            sentence_ds = test_ds.select(range(start, end))
            if config.use_bustm:
                sentence_ds = datasets.concatenate_datasets([wudao_ds, sentence_ds])
            if rank == 0:
                print(f'**********The Test_ds Range is {start} ~~ {end}**********')

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
