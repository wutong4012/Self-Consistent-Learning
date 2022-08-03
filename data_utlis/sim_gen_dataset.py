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
        if len(example['text2']) < 5:  # 最小长度设为5
            example['score'] = -1
        elif len(example['text2']) > 50:  # 最大长度设为50
            example['score'] = -2
        else:
            delta = min(len(example['text1']), len(example['text2'])) \
                - LCS(example['text1'], example['text2'])
            if delta <= 1:
                example['score'] = -3
        return example

    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()

    sim_dataset = sim_dataset.map(
        process_equal, cache_file_name=data_path+'/map_cache'+str(config.cycle))

    if rank == 0:
        cnt = sim_dataset.filter(lambda example: example['score'] == -1, 
            cache_file_name=data_path+'/short_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Short(<5) Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['score'] == -2,
            cache_file_name=data_path+'/long_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Long(>50) Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['score'] == -3,
            cache_file_name=data_path+'/bad_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Bad Sentence!**********')

    if rank == 0 and config.cycle != -1:
        torch.distributed.barrier()
    
    return sim_dataset


def load_data(config, rank, is_labeled=False, is_score=False, attri=None):
    if is_labeled:
        sim_dataset = datasets.load_from_disk(
            # '/cognitive_comp/wutong/source/sim_data/similarity_data/labeled4' + config.data_name)
            config.lab_data_path + config.data_name + '_train_ds')  # fine-tune 
        if rank > 0:
            torch.distributed.barrier()
        sim_dataset = sim_dataset.shuffle(
            config.seed + config.cycle, 
            indices_cache_file_name=config.cache_data_path+'/shuffle_cache_'+str(config.cycle))
        if rank == 0:
            torch.distributed.barrier()

    else:
        if attri == 'dis':
            if is_score:
                data_path = config.sim_data_path + '/score_cycle_{}'.format(config.cycle + 1)
            else:
                data_path = config.sim_data_path + '/trainD_cycle_{}'.format(config.cycle)

        elif attri == 'gen':
            data_path = config.sim_data_path + '/trainG_cycle_{}'.format(config.cycle)

        if rank == 0:
            print(f'Data Path: {data_path} !')
        sim_dataset = datasets.load_from_disk(data_path)

        if attri == 'gen':
            sim_dataset = preprocess_gen_data(config, rank, config.cache_data_path, sim_dataset)

    return sim_dataset


def set_dis_dataset(config, rank, part_labeled_data, generated_data, labeled_data):
    assert part_labeled_data.features.type == generated_data.features.type

    if config.dis_balance:
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
            delta_data = labeled_data.filter(
                lambda example: example['score'] == 0,
                cache_file_name=config.cache_data_path+'/lab_neg_cache_'+str(config.cycle))
        
        else:
            delta_data = labeled_data.filter(
                lambda example: example['score'] == 1,
                cache_file_name=config.cache_data_path+'/lab_pos_cache_'+str(config.cycle))
        
        if rank == 0:
            print(f'**********From Generated Data Positive Samples: {gen_pos_nums}',)
            print(f'**********From Generated Data Negtive Samples: {gen_neg_nums}', )
            torch.distributed.barrier()

        if abs(gen_pos_nums - gen_neg_nums) < delta_data.num_rows:
            delta_data = delta_data.select(range(abs(gen_pos_nums - gen_neg_nums)))
        data = datasets.concatenate_datasets(
                [part_labeled_data, generated_data, delta_data])
    
    else:
        data = datasets.concatenate_datasets([part_labeled_data, generated_data])

    if rank == 0:
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
    if not config.pretrain_dis:
        if use_gen and not use_label:
            generated_data = load_data(config, rank, is_labeled=False, attri=attri)

            if attri == 'dis':
                data = generated_data
                if rank == 0:
                    print('**********All Positive Samples: ', data.filter(
                        lambda example: example['score'] == 1,
                        cache_file_name=config.cache_data_path+'/all_pos_cache_'+str(config.cycle)).num_rows)
                    print('**********All Negtive Samples: ', data.filter(
                        lambda example: example['score'] == 0,
                        cache_file_name=config.cache_data_path+'/all_neg_cache_'+str(config.cycle)).num_rows)
                
            elif attri == 'gen':
                if rank > 0:
                    print(f'Rank {rank} waiting for main process to perform the filtering')
                    torch.distributed.barrier()

                data = generated_data.filter(
                    lambda example: example['score'] == 1,
                    cache_file_name=config.cache_data_path+'/gen2gen_cache_'+str(config.cycle))

                if rank == 0:
                    torch.distributed.barrier()

        elif use_gen and use_label:
            labeled_data = load_data(config, rank, is_labeled=True)
            generated_data = load_data(config, rank, is_labeled=False, attri=attri)
            part_labeled_data = labeled_data.select(range(generated_data.num_rows))

            if rank == 0:
                random_list = random.sample(range(part_labeled_data.num_rows), 10)
                for i in random_list:
                    print('Labeled Examples: {}'.format(part_labeled_data[i]))
                random_list = random.sample(range(generated_data.num_rows), 10)
                for i in random_list:
                    print('Generated Examples: {}'.format(generated_data[i]))

            if attri == 'dis':
                data = set_dis_dataset(
                    config, rank, part_labeled_data, generated_data, labeled_data)

            elif attri == 'gen':
                data = set_gen_dataset(
                    config, rank, part_labeled_data, generated_data)

    if config.pretrain_dis:
        train_data = datasets.load_from_disk(
            '/cognitive_comp/wutong/source/sim_data/similarity_data/labeled_train_' + config.data_name)
            # config.lab_data_path + config.data_name + '_train_ds')  # fine-tune
        train_dataset = SimGanDataset(data=train_data)
        test_data = datasets.load_from_disk(
            '/cognitive_comp/wutong/source/sim_data/similarity_data/labeled_test_' + config.data_name)
            # config.test_data_path + config.data_name)  # fine-tune
        val_dataset = SimGanDataset(data=test_data)
        
    else:
        train_dataset = SimGanDataset(data=data)
        test_data = datasets.load_from_disk(config.test_data_path + config.data_name)
        val_dataset = SimGanDataset(data=test_data)

    if rank == 0:
        print('**********Train Data: ', len(train_dataset))
        print('**********Test Data: ', len(val_dataset))
        
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
