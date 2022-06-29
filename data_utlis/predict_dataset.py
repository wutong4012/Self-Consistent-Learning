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
        real_logits = []
        for batch in dataloader:
            torch.cuda.empty_cache()
            logits = discriminator.forward(
                batch['input_ids'].cuda(), None)
            for idx in range(logits.size(0)):
                logits = torch.softmax(logits, dim=1)
                real_logits.append(logits[idx][1].cpu())
    discriminator.to('cpu')
    logits_list = torch.softmax(
        torch.tensor(real_logits), dim=0).numpy().tolist()

    multi_logits = []
    for idx in range(raw_dataset.num_rows):
        multi_logits.append(logits_list[idx] * raw_dataset[idx]['-ppl'])
    multi_logits = torch.softmax(
        torch.tensor(multi_logits), dim=0).numpy().tolist()
    
    scores = []
    for idx in range(len(multi_logits)):
        if multi_logits[idx] >= config.gen_threshold * (1 / len(multi_logits)):
            scores.append(1)
        else:
            scores.append(0)
    if rank == 0:
        print(f'**********There are {scores.count(1)} Samples to be selected!**********')

    return {
        'text1': raw_dataset['text1'],
        'text2': raw_dataset['text2'],
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
        real_ppl_list.append(-output_dict['ppl_list'][idx])  # 加上负号，使其越大越好

    ppl_list = torch.softmax(
        torch.tensor(real_ppl_list), dim=0).numpy().tolist()
    raw_dataset = Dataset.from_dict({
        'text1': raw_text,
        'text2': sim_text,
        '-ppl': ppl_list,
    })
    gen_ds_dict = multiply_pre_score(config, raw_dataset, rank)
    gen_dataset = Dataset.from_dict(gen_ds_dict, features=feats)

    if rank == 0:
        new_data_path = config.gen_data_path + f'_cycle_{config.cycle + 1}'
        if not os.path.exists(new_data_path):
            os.makedirs(new_data_path)
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

        elif item[0] >= config.dis_threshold:
            scores.append(0)
            text1.append(dis_output_dict['text1'][idx])
            text2.append(dis_output_dict['text2'][idx])

    dis_dataset = Dataset.from_dict({
        'text1': text1,
        'text2': text2,
        'score': scores,
    }, features=feats)

    if rank == 0:
        new_data_path = config.score_data_path + f'_cycle_{config.cycle + 1}'
        if not os.path.exists(new_data_path):
            os.makedirs(new_data_path)
        print(f'**********Score Data Samples is {dis_dataset.num_rows}**********')
        dis_dataset.save_to_disk(new_data_path)
        print('**********Score Data: done!!!**********')


def gen_pred_collate(batch_data, gen_tokenizer, config):
    input_ids, length_list = [], []
    for item in batch_data:
        item = item['sentence_list']
        if item is None or item == []:
            continue

        # 每段话只随机选一条句子
        random_num = random.sample(range(len(item)), 1)[0]
        while len(item[random_num]) < 10 or len(item[random_num]) > 100:
            random_num = random.sample(range(len(item)), 1)[0]

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

        wudao_ds = load_data(config, rank, is_wudao=True)
        predict_dataset = SimGanDataset(wudao_ds)

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