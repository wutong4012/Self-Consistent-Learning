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


def load_data(config, rank, is_labeled=False, is_wudao=False,
              is_score=False, attri=None):
    if is_wudao:
        cache_dict_paths = glob.glob(config.wudao_data_path + '/*')
        cache_dict_paths = cache_dict_paths[(config.cycle+1)*3:(config.cycle+2)*3]
        wudao_ds_list = []
        for path in cache_dict_paths:
            wudao_ds_list.append(datasets.load_from_disk(path))
        wudao_ds = datasets.concatenate_datasets(wudao_ds_list)
        return wudao_ds

    if is_labeled:  # 1590792 -> 1488200 -> 1391008
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
            if min(len(example['text1']), len(example['text2'])) <= 10:  # 最小长度设为10
                example['score'] = -1
            else:
                delta = min(len(example['text1']), len(
                    example['text2'])) - LCS(example['text1'], example['text2'])
                if delta <= 1:
                    example['score'] = -2
                elif delta <= 2 and attri == 'gen':
                    example['score'] = -3
            return example
        if rank > 0:
            print(f'Rank {rank} waiting for main process to perform the mapping')
            torch.distributed.barrier()
        sim_dataset = sim_dataset.map(
            process_equal, cache_file_name=data_path+'/map_cache')
        if rank == 0 and config.cycle != -1:
            torch.distributed.barrier()

        if attri == 'dis':
            if rank == 0:
                cnt = sim_dataset.filter(lambda example: example['score'] == -1,
                                         cache_file_name=data_path+'/short_cache').num_rows
                print(f'There are {cnt} Short(<=10) Sentence!')
                cnt = sim_dataset.filter(lambda example: example['score'] == -2,
                                         cache_file_name=data_path+'/bad_cache').num_rows
                print(f'There are {cnt} Bad Sentence!')

            if rank > 0:
                print(f'Rank {rank} waiting for main process to perform the filtering')
                torch.distributed.barrier()
            sim_dataset = sim_dataset.filter(lambda example: example['score'] != -1,
                                             cache_file_name=data_path+'/del_short_cache')
            sim_dataset = sim_dataset.filter(lambda example: example['score'] != -2,
                                             cache_file_name=data_path+'/del_bad_cache')
            if rank == 0 and config.cycle != -1:
                torch.distributed.barrier()

        elif attri == 'gen':
            if rank == 0:
                cnt = sim_dataset.filter(lambda example: example['score'] == -3,
                                         cache_file_name=data_path+'/equal_cache').num_rows
                print(f'There are {cnt} Equal Sentence!')

    return sim_dataset


def set_dis_dataset(config, rank, start, end, 
                    part_labeled_data, generated_data, labeled_data):
    assert part_labeled_data.features.type == generated_data.features.type
    if config.cycle <= config.gen_anti_cyle:
        if rank > 0:
            print(f'Rank {rank} waiting for main process to perform the filtering')
            torch.distributed.barrier()
        def filter_fn(example, idx):
            return ((idx <= start) or (idx >= end)) and (example['score'] == 1)
        positived_data = labeled_data.filter(
            filter_fn, with_indices=True,
            cache_file_name=config.cache_data_path+'/lab_pos_cache_'+str(config.cycle))
        if rank == 0:
            torch.distributed.barrier()

        positived_data = positived_data.select(range(generated_data.num_rows))
        data = datasets.concatenate_datasets(
            [part_labeled_data, generated_data, positived_data])

    else:
        if rank > 0:
            print(f'Rank {rank} waiting for main process to perform the filtering')
            torch.distributed.barrier()
        def filter_fn(example, idx):
            return ((idx <= start) or (idx >= end)) and (example['score'] == 0)
        negtived_data = labeled_data.filter(
            filter_fn, with_indices=True,
            cache_file_name=config.cache_data_path+'/lab_neg_cache_'+str(config.cycle))
        if rank == 0:
            torch.distributed.barrier()

        negtived_data = negtived_data.select(range(generated_data.num_rows))
        data = datasets.concatenate_datasets(
            [part_labeled_data, generated_data, negtived_data])

    if rank == 0:
        print('From Generated Data Positive Samples: ', generated_data.filter(
            lambda example: example['score'] == 1,
            cache_file_name=config.cache_data_path+'/gen_pos_cache_'+str(config.cycle)).num_rows)
        print('From Generated Data Negtive Samples: ', generated_data.filter(
            lambda example: example['score'] == 0,
            cache_file_name=config.cache_data_path+'/gen_neg_cache_'+str(config.cycle)).num_rows)
        print('All Positive Samples: ', data.filter(
            lambda example: example['score'] == 1,
            cache_file_name=config.cache_data_path+'/all_pos_cache_'+str(config.cycle)).num_rows)
        print('All Negtive Samples: ', data.filter(
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
    data = datasets.concatenate_datasets(
        [part_labeled_data, generated_data])

    if rank == 0:
        print(f'All Gen-Data Samples is {data.num_rows}')
        print(f'{generated_data.num_rows} Filter Samples From Generated Data')
        
    return data


def set_dataset(config, use_label, use_gen, attri, rank):
    if use_label and not use_gen:
        data = load_data(config, rank, is_labeled=True)
    elif use_gen and not use_label:
        data = load_data(config, rank, is_labeled=False, attri=attri)
    elif use_gen and use_label:
        labeled_data = load_data(config, rank, is_labeled=True)
        generated_data = load_data(config, rank, is_labeled=False, attri=attri)

        start, end = (config.cycle * generated_data.num_rows * 2), (
            (config.cycle + 1) * generated_data.num_rows * 2)
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

    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the spliting')
        torch.distributed.barrier()
    data = data.train_test_split(
        train_size=0.8, test_size=0.2, seed=config.seed,
        train_indices_cache_file_name=config.cache_data_path+'/train'+attri+str(config.cycle),
        test_indices_cache_file_name=config.cache_data_path+'/test'+attri+str(config.cycle))
    if rank == 0:
        torch.distributed.barrier()
    train_dataset = SimGanDataset(data=data['train'])
    val_dataset = SimGanDataset(data=data['test'])

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
