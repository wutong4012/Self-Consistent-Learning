import glob
import random
import datasets
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.utilities import rank_zero_only

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


def load_data(config, is_labeled=False, is_wudao=False, is_score=False, attri=None):
    if is_wudao:
        cache_dict_paths = glob.glob(config.wudao_data_path + '/*')
        data_path = cache_dict_paths[config.cycle]
        wudao_ds = datasets.load_from_disk(data_path)
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
        print(f'Data Path: {data_path} !')
        sim_dataset = datasets.load_from_disk(data_path)
        
        def LCS(str1, str2):
            len_str1 = len(str1)
            len_str2 = len(str2)
            record = [[0 for i in range(len_str2 + 1)] for j in range(len_str1 + 1)]
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
                delta = min(len(example['text1']), len(example['text2'])) - LCS(example['text1'], example['text2'])
                if delta <= 1:
                    example['score'] = -2
                elif delta <= 2 and attri == 'gen':
                    example['score'] = -3
            return example
        sim_dataset = sim_dataset.map(process_equal)
        
        if attri == 'dis':
            cnt = sim_dataset.filter(lambda example: example['score'] == -1).num_rows
            print(f'There are {cnt} Short(<=10) Sentence!')
            cnt = sim_dataset.filter(lambda example: example['score'] == -2).num_rows
            print(f'There are {cnt} Bad Sentence!')
            sim_dataset = sim_dataset.filter(lambda example: example['score'] != -1)
            sim_dataset = sim_dataset.filter(lambda example: example['score'] != -2)
            
        elif attri == 'gen':
            cnt = sim_dataset.filter(lambda example: example['score'] == -3).num_rows
            print(f'There are {cnt} Equal Sentence!')

    return sim_dataset


def set_dataset(config, use_label, use_gen, attri=None):
    if use_label and not use_gen:
        data = load_data(config, is_labeled=True)
    elif use_gen and not use_label:
        data = load_data(config, is_labeled=False, attri=attri)
    elif use_gen and use_label:
        labeled_data = load_data(config, is_labeled=True)
        generated_data = load_data(config, is_labeled=False, attri=attri)

        start, end = (config.cycle * generated_data.num_rows * 2), (
            (config.cycle + 1) * generated_data.num_rows * 2)
        part_labeled_data = labeled_data.select(range(start, end))
        
        random_list = random.sample(range(part_labeled_data.num_rows), 10)
        for i in random_list:
            print('Labeled Examples: {}'.format(part_labeled_data[i]))
        random_list = random.sample(range(generated_data.num_rows), 10)
        for i in random_list:
            print('Generated Examples: {}'.format(generated_data[i]))
        
        if attri == 'dis':
            assert part_labeled_data.features.type == generated_data.features.type
            if config.cycle < config.gen_anti_cyle:
                def filter_fn(example, idx):
                    return ((idx <= start) or (idx >= end)) and (example['score'] == 1)
                positived_data = labeled_data.filter(filter_fn, with_indices=True)
                positived_data = positived_data.select(range(generated_data.num_rows))
                data = datasets.concatenate_datasets(
                    [part_labeled_data, generated_data, positived_data])
            else:
                def filter_fn(example, idx):
                    return ((idx <= start) or (idx >= end)) and (example['score'] == 0)
                negtived_data = labeled_data.filter(filter_fn, with_indices=True)
                negtived_data = negtived_data.select(range(generated_data.num_rows))
                data = datasets.concatenate_datasets(
                    [part_labeled_data, generated_data, negtived_data])
            print('From Generated Data Positive Samples: ', generated_data.filter(
                lambda example: example['score'] == 1).num_rows)
            print('From Generated Data Negtive Samples: ', generated_data.filter(
                lambda example: example['score'] == 0).num_rows)
            print('All Positive Samples: ', data.filter(
                lambda example: example['score'] == 1).num_rows)
            print('All Negtive Samples: ', data.filter(
                lambda example: example['score'] == 0).num_rows)
            
        elif attri == 'gen':
            part_labeled_data = part_labeled_data.filter(lambda example: example['score'] == 1)
            generated_data = generated_data.filter(lambda example: example['score'] == 1)
            data = datasets.concatenate_datasets(
                [part_labeled_data, generated_data])
            print(f'All Gen-Data Samples is {data.num_rows}')
            print(f'{generated_data.num_rows} Filter Samples From Generated Data')

    data = data.train_test_split(
        train_size=0.8, test_size=0.2, seed=config.seed)
    train_dataset = SimGanDataset(data=data['train'])
    val_dataset = SimGanDataset(data=data['test'])

    return train_dataset, val_dataset


def create_dataloader(config, dataset, tokenizer, attri='gen', shuffle=True):
    if attri == 'dis':
        batch_size = 160

        def collate_fn(batch_data):
            return discriminator_collate_fn(
                batch_data, tokenizer, config.dis_batch_size, is_train=shuffle)

    elif attri == 'gen':
        batch_size = 30

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
