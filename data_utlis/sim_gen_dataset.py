import glob
import random

import torch
import datasets
from torch.utils.data import DataLoader, Dataset

from data_utlis.sim_data_collate import (discriminator_collate_fn,
                                         generator_collate_fn)


class SimGanDataset(Dataset):
    """
        labeled Data(datasets): text1(str), text2(str), score(int8) 
        Generated Data(datasets): text1(str), text2(str), score(int8) 
    """

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.num_rows


def preprocess_gen_data(config, rank, data_path, sim_dataset):

    def LCS(str1, str2):
        len_str1 = len(str1)
        len_str2 = len(str2)
        record = [[0 for i in range(len_str2 + 1)]
                  for j in range(len_str1 + 1)]
        for i in range(len_str1):
            for j in range(len_str2):
                if str1[i] == str2[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                elif record[i + 1][j] > record[i][j + 1]:
                    record[i + 1][j + 1] = record[i + 1][j]
                else:
                    record[i + 1][j + 1] = record[i][j + 1]
        return record[-1][-1]

    def process_equal(example):
        if min(len(example['text1']), len(example['text2'])) < 10:  # 最小长度设为10
            example['score'] = -1
        elif max(len(example['text1']), len(example['text2'])) > 100:  # 最大长度设为100
            example['score'] = -2
        else:
            delta = min(len(example['text1']), len(
                example['text2'])) - LCS(example['text1'], example['text2'])
            if delta <= 1:
                example['score'] = -3
            elif delta <= 2:
                example['score'] = -4
        return example

    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()

    sim_dataset = sim_dataset.map(
        process_equal, cache_file_name=data_path+'/map_cache')

    if rank == 0:
        cnt = sim_dataset.filter(lambda example: example['score'] == -1,
                                 cache_file_name=data_path+'/short_cache').num_rows
        print(f'**********There are {cnt} Short(<10) Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['score'] == -2,
                                 cache_file_name=data_path+'/long_cache').num_rows
        print(f'**********There are {cnt} Long(>100) Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['score'] == -3,
                                 cache_file_name=data_path+'/bad_cache').num_rows
        print(f'**********There are {cnt} Bad Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['score'] == -4,
                                 cache_file_name=data_path+'/equal_cache').num_rows
        print(f'**********There are {cnt} Equal Sentence!**********')

    if rank == 0 and config.cycle != -1:
        torch.distributed.barrier()
    
    return sim_dataset


def load_data(config, rank, is_labeled=False, is_wudao=False,
              is_score=False, attri=None):
    if is_wudao:
        cache_dict_paths = glob.glob(config.wudao_data_path + '/*')
        cache_dict_paths = cache_dict_paths[
            (config.cycle + 1) * 2 : (config.cycle + 2) * 2]

        wudao_ds_list = []
        for path in cache_dict_paths:
            wudao_ds_list.append(datasets.load_from_disk(path))
        wudao_ds = datasets.concatenate_datasets(wudao_ds_list)

        return wudao_ds

    if is_labeled:
        cache_dict_paths = glob.glob(config.lab_data_path + '/*')

        sim_ds_list = []
        for path in cache_dict_paths:
            sim_ds_list.append(datasets.load_from_disk(path))
        sim_dataset = datasets.concatenate_datasets(sim_ds_list)

    else:
        if attri == 'dis':
            if is_score:
                data_path = config.gen_data_path + '_cycle_{}'.format(config.cycle + 1)
            else:
                data_path = config.gen_data_path + '_cycle_{}'.format(config.cycle)

        elif attri == 'gen':
            data_path = config.score_data_path + '_cycle_{}'.format(config.cycle)

        if rank == 0:
            print(f'Data Path: {data_path} !')
        sim_dataset = datasets.load_from_disk(data_path)

        if attri == 'gen':
            sim_dataset = preprocess_gen_data(config, rank, data_path, sim_dataset)

    return sim_dataset


def set_dis_dataset(config, rank, start, end,
                    part_labeled_data, generated_data, labeled_data):
    assert part_labeled_data.features.type == generated_data.features.type

    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the filtering')
        torch.distributed.barrier()
    
    gen_pos_nums = generated_data.filter(
        lambda example: example['score'] == 1, 
        cache_file_name=config.cache_data_path+'/gen_pos_cache_'+str(config.cycle)
    ).num_rows
    gen_neg_nums = generated_data.filter(
        lambda example: example['score'] == 0,
        cache_file_name=config.cache_data_path+'/gen_neg_cache_'+str(config.cycle)
    ).num_rows

    if gen_pos_nums > gen_neg_nums:
        def filter_fn(example, idx):
            return ((idx <= start) or (idx >= end)) and (example['score'] == 0)
        delta_data = labeled_data.filter(
            filter_fn, with_indices=True,
            cache_file_name=config.cache_data_path+'/lab_neg_cache_'+str(config.cycle))
    
    else:
        def filter_fn(example, idx):
            return ((idx <= start) or (idx >= end)) and (example['score'] == 1)
        delta_data = labeled_data.filter(
            filter_fn, with_indices=True,
            cache_file_name=config.cache_data_path+'/lab_pos_cache_'+str(config.cycle))
    
    if rank == 0:
        torch.distributed.barrier()

    delta_data = delta_data.select(range(abs(gen_pos_nums - gen_neg_nums)))
    data = datasets.concatenate_datasets(
            [part_labeled_data, generated_data, delta_data])

    if rank == 0:
        print(f'**********From Generated Data Positive Samples: {gen_pos_nums}',)
        print(f'**********From Generated Data Negtive Samples: {gen_neg_nums}', )
        print('**********All Positive Samples: ', data.filter(
            lambda example: example['score'] == 1,
            cache_file_name=config.cache_data_path+'/all_pos_cache_'+str(config.cycle)).num_rows)
        print('**********All Negtive Samples: ', data.filter(
            lambda example: example['score'] == 0,
            cache_file_name=config.cache_data_path+'/all_neg_cache_'+str(config.cycle)).num_rows)

    return data


def set_gen_dataset(config, rank, part_labeled_data, generated_data):
    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the filtering')
        torch.distributed.barrier()

    part_labeled_data = part_labeled_data.filter(
        lambda example: example['score'] == 1,
        cache_file_name=config.cache_data_path+'/lab2gen_cache_'+str(config.cycle))
    generated_data = generated_data.filter(
        lambda example: example['score'] == 1,
        cache_file_name=config.cache_data_path+'/gen2gen_cache_'+str(config.cycle))

    if rank == 0:
        torch.distributed.barrier()

    data = datasets.concatenate_datasets([part_labeled_data, generated_data])

    if rank == 0:
        print(f'**********All Gen-Data Samples is {data.num_rows}**********')
        print(f'**********{generated_data.num_rows} Filter Samples From Generated Data**********')

    return data


def set_dataset(config, use_label, use_gen, attri, rank):
    if use_label and not use_gen:
        data = load_data(config, rank, is_labeled=True)

    elif use_gen and not use_label:
        data = load_data(config, rank, is_labeled=False, attri=attri)

    elif use_gen and use_label:
        labeled_data = load_data(config, rank, is_labeled=True)
        generated_data = load_data(config, rank, is_labeled=False, attri=attri)

        start, end = (config.cycle * generated_data.num_rows), (
            (config.cycle + 1) * generated_data.num_rows)
        part_labeled_data = labeled_data.select(range(start, end))

        if rank == 0:
            random_list = random.sample(range(part_labeled_data.num_rows), 10)
            for i in random_list:
                print('Labeled Examples: {}'.format(part_labeled_data[i]))
            random_list = random.sample(range(generated_data.num_rows), 10)
            for i in random_list:
                print('Generated Examples: {}'.format(generated_data[i]))

        if attri == 'dis':
            data = set_dis_dataset(
                config, rank, start, end, part_labeled_data, generated_data, labeled_data)

        elif attri == 'gen':
            data = set_gen_dataset(
                config, rank, part_labeled_data, generated_data)

    train_dataset = SimGanDataset(data=data)
    test_data = datasets.load_from_disk(config.test_data_path)
    val_dataset = SimGanDataset(data=test_data)

    torch.distributed.barrier()
    return train_dataset, val_dataset


def create_dataloader(config, dataset, tokenizer, attri='gen', shuffle=True):
    if attri == 'dis':
        batch_size = config.dis_batch_size

        def collate_fn(batch_data):
            return discriminator_collate_fn(
                batch_data, tokenizer, is_train=shuffle)

    elif attri == 'gen':
        batch_size = config.gen_big_batch_size

        def collate_fn(batch_data):
            return generator_collate_fn(
                batch_data, tokenizer, config.gen_batch_size, is_train=shuffle)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader
