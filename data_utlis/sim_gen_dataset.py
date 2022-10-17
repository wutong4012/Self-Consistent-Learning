import random

import torch
import datasets
import numpy as np
from torch.utils.data import DataLoader, Dataset

from data_utlis.sim_data_collate import (discriminator_collate_fn,
                                         generator_collate_fn,
                                         generator_en_collate_fn)


def preprocess(sample):
    sample['text1'] = sample['sentence1']
    sample['text2'] = ''
    sample['question'] = '怎么理解这段话？'
    sample['choice'] = ["不能理解为："+sample['sentence2'],
                        "可以理解为："+sample['sentence2']]
    return sample


def get_att_mask(attention_mask, label_idx, question_len):
    max_length = len(attention_mask)
    attention_mask = np.array(attention_mask)
    attention_mask = np.tile(attention_mask[None, :], (max_length, 1))

    zeros = np.zeros(
        shape=(label_idx[-1]-question_len, label_idx[-1]-question_len))
    
    attention_mask[question_len:label_idx[-1],
                   question_len:label_idx[-1]] = zeros

    for i in range(len(label_idx)-1):
        label_token_length = label_idx[i+1]-label_idx[i]
        if label_token_length <= 0:
            print('label_idx', label_idx)
            print('question_len', question_len)
            continue
        ones = np.ones(shape=(label_token_length, label_token_length))
        attention_mask[label_idx[i]:label_idx[i+1],
                       label_idx[i]:label_idx[i+1]] = ones

    return attention_mask


def get_position_ids(label_idx, max_length, question_len):
    question_position_ids = np.arange(question_len)
    label_position_ids = np.arange(question_len, label_idx[-1])
    for i in range(len(label_idx)-1):
        label_position_ids[label_idx[i]-question_len:label_idx[i+1]-question_len] = np.arange(
            question_len, question_len+label_idx[i+1]-label_idx[i])
    max_len_label = max(label_position_ids)
    text_position_ids = np.arange(
        max_len_label+1, max_length+max_len_label+1-label_idx[-1])
    position_ids = list(question_position_ids) + \
        list(label_position_ids)+list(text_position_ids)
    if max_length <= 512:
        return position_ids[:max_length]
    else:
        for i in range(512, max_length):
            if position_ids[i] > 511:
                position_ids[i] = 511
        return position_ids[:max_length]


def random_masking(token_ids, mask_rate, mask_start_idx, max_length, tokenizer=None):
    rands = np.random.random(len(token_ids))
    source, target, mask_pos = [], [], []

    # 删除-CLS SEP id
    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    for i, (r, t) in enumerate(zip(rands, token_ids)):
        if i < mask_start_idx:
            source.append(t)
            target.append(-100)
            continue
        if t == cls_id or t == sep_id:
            source.append(t)
            target.append(-100)
            continue

        if r < mask_rate * 0.8:
            source.append(mask_id)
            target.append(t)
            mask_pos.append(i)
        elif r < mask_rate * 0.9:
            source.append(t)
            target.append(t)
            mask_pos.append(i)
        elif r < mask_rate:
            source.append(np.random.choice(
                len(tokenizer.get_vocab().keys()) - 2) + 1)
            target.append(t)
            mask_pos.append(i)
        else:
            source.append(t)
            target.append(-100)

    while len(source) < max_length:
        source.append(0)
        target.append(-100)
    return source[:max_length], target[:max_length], mask_pos


class SimGanDataset(Dataset):
    """
        labeled Data(datasets): text1(str), text2(str), label(int8) 
        Generated Data(datasets): text1(str), text2(str), label(int8) 
    """

    def __init__(self, data, tokenizer=None, predict=False, is_gen=False, test=False) -> None:
        super().__init__()
        self.data = data
        self.test = test
        self.is_gen = is_gen
        self.predict = predict
        self.tokenizer = tokenizer
        
        if tokenizer != None:
            self.yes_token_id = tokenizer.encode("是")[1]
            self.no_token_id = tokenizer.encode("非")[1]

    def __getitem__(self, index):
        item = self.data[index]
        if self.is_gen:
            return item

        if self.predict:
            one_data = self.encode(item, False, True)
        elif self.test:
            one_data = self.encode(item, False, False)
        else:
            one_data = self.encode(item, True, False)

        return one_data

    def __len__(self):
        return self.data.num_rows

    def encode(self, item, used_mask, unlabeled):
        if not unlabeled:
            item['label'] = int(item['label'])
            item['answer'] = item['choice'][item['label']]
        else:
            item['label'] = None

        if item['text2'] != '':
            text1 = '[MASK]' + '[MASK]'.join(
                item['choice']) + '[SEP]' + item['question'] + '[SEP]' + item['text1']
            encode_dict = self.tokenizer.encode_plus(text1,
                                                     max_length=512,
                                                     padding="longest",
                                                     truncation=True
                                                     )
        else:
            text1 = '[MASK]' + '[MASK]'.join(
                item['choice']) + '[SEP]'+item['question'] + '[SEP]' + item['text1']
            encode_dict = self.tokenizer.encode_plus(text1,
                                                     max_length=512,
                                                     padding="longest",
                                                     truncation=True
                                                     )

        encode_sent = encode_dict['input_ids']
        token_type_ids = encode_dict['token_type_ids']
        attention_mask = encode_dict['attention_mask']

        question_len = 1
        label_idx = [question_len]
        for choice in item['choice']:
            cur_mask_idx = label_idx[-1] + \
                len(self.tokenizer.encode(choice, add_special_tokens=False))+1
            label_idx.append(cur_mask_idx)

        encoded_len = len(encode_dict["input_ids"])
        zero_len = len(encode_dict["input_ids"]) - \
            question_len - ((label_idx[-1]-label_idx[0]+1))
        token_type_ids = [0]*question_len+[1] * \
            (label_idx[-1]-label_idx[0]+1)+[0]*zero_len

        attention_mask = get_att_mask(attention_mask, label_idx, question_len)

        position_ids = get_position_ids(label_idx, encoded_len, question_len)

        clslabels_mask = np.zeros(shape=(len(encode_sent),))
        clslabels_mask[label_idx[:-1]] = 10000
        clslabels_mask = clslabels_mask-10000

        mlmlabels_mask = np.zeros(shape=(len(encode_sent),))
        mlmlabels_mask[label_idx[0]] = 1

        if used_mask:
            mask_rate = 0.1*np.random.choice(4, p=[0.3, 0.3, 0.25, 0.15])
            source, target, _ = random_masking(token_ids=encode_sent, mask_rate=mask_rate,
                                               mask_start_idx=label_idx[-1], max_length=encoded_len,
                                               tokenizer=self.tokenizer)
        else:
            source, target = encode_sent[:], encode_sent[:]

        source = np.array(source)
        target = np.array(target)
        source[label_idx[:-1]] = self.tokenizer.mask_token_id
        target[label_idx[:-1]] = self.no_token_id
        if unlabeled:
            rand_idx = label_idx[0]
            target[rand_idx] = self.yes_token_id
            clslabels = label_idx[0]
        else:
            target[label_idx[item['label']]] = self.yes_token_id
            clslabels = label_idx[item['label']]

        end_token = ["[SEP]"]
        end_id = self.tokenizer.convert_tokens_to_ids(end_token)[0]
        seq_actual_len = len(source) - list(source[::-1]).index(end_id)

        one_data = {
            "input_ids": torch.tensor(source).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "position_ids": torch.tensor(position_ids).long(),
            "mlmlabels": torch.tensor(target).long(),
            "clslabels": torch.tensor(clslabels).long(),
            # 'label_idx': torch.tensor(label_idx).long(),
            "clslabels_mask": torch.tensor(clslabels_mask).float(),
            "mlmlabels_mask": torch.tensor(mlmlabels_mask).float(),
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'label': item['label']
        }

        return one_data


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


def preprocess_gen_data(config, rank, data_path, sim_dataset):
    def process_equal(example):
        if len(example['sentence2']) < 2:  # 最小长度设为2
            example['label'] = -1
        elif len(example['sentence2']) > 80:  # 最大长度设为80
            example['label'] = -2
        else:
            delta = min(len(example['sentence1']), len(example['sentence2'])) \
                - LCS(example['sentence1'], example['sentence2'])
            if delta <= 1:
                example['label'] = -3
        return example

    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()

    sim_dataset = sim_dataset.map(
        process_equal, cache_file_name=data_path+'/map_cache'+str(config.cycle))

    if rank == 0:
        cnt = sim_dataset.filter(lambda example: example['label'] == -1,
                                 cache_file_name=data_path+'/short_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Short(<2) Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['label'] == -2,
                                 cache_file_name=data_path+'/long_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Long(>50) Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['label'] == -3,
                                 cache_file_name=data_path+'/bad_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Bad Sentence!**********')

    if rank == 0 and config.cycle != -1:
        torch.distributed.barrier()

    return sim_dataset


def preprocess_gen_data_en(config, rank, data_path, sim_dataset):
    def process_equal(example):
        if len(example['text1'].split(' ')) > 5 and len(example['text2'].split(' ')) < 5:  # 最小长度设为5
            example['label'] = -2
        else:
            text1, text2 = example['text1'].lower(), example['text2'].lower()
            delta = min(len(text1), len(text2)) - LCS(text1, text2)
            if delta <= 3:
                example['label'] = -1
        return example

    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the mapping')
        torch.distributed.barrier()

    sim_dataset = sim_dataset.map(
        process_equal, cache_file_name=data_path+'/map_cache_en'+str(config.cycle))

    if rank == 0:
        cnt = sim_dataset.filter(lambda example: example['label'] == -1,
                                 cache_file_name=data_path+'/short_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Equal Sentence!**********')
        cnt = sim_dataset.filter(lambda example: example['label'] == -2,
                                 cache_file_name=data_path+'/long_cache'+str(config.cycle)).num_rows
        print(f'**********There are {cnt} Short(<5) Sentence!**********')

    if rank == 0 and config.cycle != -1:
        torch.distributed.barrier()

    return sim_dataset


def load_data(config, rank, is_labeled=False, is_score=False, attri=None):
    if is_labeled:
        sim_dataset = datasets.Dataset.from_json(
            '/cognitive_comp/wutong/source/sim_data/raw_data/bustm/train_0.json')
        sim_dataset = sim_dataset.remove_columns(['id'])
        feats = datasets.Features({"sentence1": datasets.Value('string'),
                                    "sentence2": datasets.Value('string'),
                                    "label": datasets.Value('int8')})
        sim_dataset = sim_dataset.map(features=feats)

    else:
        if attri == 'dis':
            if is_score:
                data_path = config.sim_data_path + \
                    '/score_cycle_{}'.format(config.cycle + 1)
            else:
                data_path = config.sim_data_path + \
                    '/trainD_cycle_{}'.format(config.cycle)

        elif attri == 'gen' or attri == 'gen_en':
            data_path = config.sim_data_path + \
                '/trainG_cycle_{}'.format(config.cycle)

        if rank == 0:
            print(f'Data Path: {data_path} !')
        sim_dataset = datasets.load_from_disk(data_path)

        if attri == 'gen':
            sim_dataset = preprocess_gen_data(
                config, rank, config.cache_data_path, sim_dataset)
        elif attri == 'gen_en':
            sim_dataset = preprocess_gen_data_en(
                config, rank, config.cache_data_path, sim_dataset)

    return sim_dataset


def set_dis_dataset(config, rank, labeled_data, generated_data):
    assert labeled_data.features.type == generated_data.features.type

    data = datasets.concatenate_datasets([labeled_data, generated_data])

    if rank == 0:
        print('**********All Positive Samples: ', data.filter(
            lambda example: example['label'] == 1,
            cache_file_name=config.cache_data_path+'/all_pos_cache_'+str(config.cycle)).num_rows)
        print('**********All Negtive Samples: ', data.filter(
            lambda example: example['label'] == 0,
            cache_file_name=config.cache_data_path+'/all_neg_cache_'+str(config.cycle)).num_rows)

    return data


def set_gen_dataset(config, rank, labeled_data, generated_data):
    if rank > 0:
        print(f'Rank {rank} waiting for main process to perform the filtering')
        torch.distributed.barrier()

    labeled_data = labeled_data.filter(
        lambda example: example['label'] == 1,
        cache_file_name=config.cache_data_path+'/lab2gen_cache_'+str(config.cycle))
    generated_data = generated_data.filter(
        lambda example: example['label'] == 1,
        cache_file_name=config.cache_data_path+'/gen2gen_cache_'+str(config.cycle))

    if rank == 0:
        torch.distributed.barrier()

    data = datasets.concatenate_datasets([labeled_data, generated_data])

    if rank == 0:
        print(f'**********All Gen-Data Samples is {data.num_rows}**********')
        print(
            f'**********{generated_data.num_rows} Filter Samples From Generated Data**********')

    return data


def set_dataset(config, use_label, use_gen, attri, rank, dis_tokenizer=None):
    if not config.pretrain_dis and not config.pretrain_gen:
        if use_gen and not use_label:
            generated_data = load_data(
                config, rank, is_labeled=False, attri=attri)

            if attri == 'dis':
                data = generated_data
                if rank == 0:
                    print('**********All Positive Samples: ', data.filter(
                        lambda example: example['label'] == 1,
                        cache_file_name=config.cache_data_path+'/all_pos_cache_'+str(config.cycle)).num_rows)
                    print('**********All Negtive Samples: ', data.filter(
                        lambda example: example['label'] == 0,
                        cache_file_name=config.cache_data_path+'/all_neg_cache_'+str(config.cycle)).num_rows)

            elif attri == 'gen' or attri == 'gen_en':
                if rank > 0:
                    print(
                        f'Rank {rank} waiting for main process to perform the filtering')
                    torch.distributed.barrier()

                data = generated_data.filter(
                    lambda example: example['label'] == 1,
                    cache_file_name=config.cache_data_path+'/gen2gen_cache_'+str(config.cycle))

                if rank == 0:
                    torch.distributed.barrier()

        elif use_gen and use_label:
            labeled_data = load_data(config, rank, is_labeled=True)
            generated_data = load_data(
                config, rank, is_labeled=False, attri=attri)

            if rank == 0:
                random_list = random.sample(range(labeled_data.num_rows), 10)
                for i in random_list:
                    print('Labeled Examples: {}'.format(labeled_data[i]))
                random_list = random.sample(range(generated_data.num_rows), 10)
                for i in random_list:
                    print('Generated Examples: {}'.format(generated_data[i]))

            if attri == 'dis':
                data = set_dis_dataset(
                    config, rank, labeled_data, generated_data)

            elif attri == 'gen' or attri == 'gen_en':
                data = set_gen_dataset(
                    config, rank, labeled_data, generated_data)

    if config.pretrain_dis:
        if config.zero_shot:
            train_data = datasets.load_from_disk(
                '/cognitive_comp/wutong/source/sim_data/similarity_data/labeled_train_' + config.data_name)
            test_data = datasets.load_from_disk(
                '/cognitive_comp/wutong/source/sim_data/similarity_data/labeled_test_' + config.data_name)
        else:
            train_data = datasets.load_from_disk(
                config.lab_data_path + config.data_name + '_train_ds')
            test_data = datasets.load_from_disk(
                config.test_data_path + config.data_name)
        train_dataset = SimGanDataset(data=train_data)
        val_dataset = SimGanDataset(data=test_data)

    elif config.pretrain_gen:
        train_data = datasets.load_from_disk(
            '/cognitive_comp/wutong/source/sim_data/similarity_data_en/pre_train')
        train_dataset = SimGanDataset(data=train_data)
        test_data = datasets.load_from_disk(
            '/cognitive_comp/wutong/source/sim_data/similarity_data_en/pre_val')
        val_dataset = SimGanDataset(data=test_data)

    else:
        test_data = datasets.Dataset.from_json(
            '/cognitive_comp/wutong/source/sim_data/raw_data/bustm/test_public.json')
        
        if attri == 'gen':
            train_dataset = SimGanDataset(data=data, is_gen=True)
            val_dataset = SimGanDataset(
                data=test_data, tokenizer=None, predict=True, is_gen=True)
        
        elif attri == 'dis':
            if rank > 0:
                print(f'Rank {rank} waiting for main process to perform the mapping')
                torch.distributed.barrier()
            data = data.map(preprocess, cache_file_name=config.cache_data_path+'/train_cache'+str(config.cycle))
            test_data = test_data.map(preprocess, 
                                    cache_file_name=config.cache_data_path+'/test_cache'+str(config.cycle))
            if rank == 0:
                torch.distributed.barrier()

            train_dataset = SimGanDataset(data=data, tokenizer=dis_tokenizer)
            val_dataset = SimGanDataset(
                data=test_data, tokenizer=dis_tokenizer, predict=True)

    if rank == 0:
        print('**********Train Data: ', len(train_dataset))
        print('**********Test Data: ', len(val_dataset))

    torch.distributed.barrier()
    return train_dataset, val_dataset


def create_dataloader(config, dataset, tokenizer, attri=None, shuffle=True):
    if attri == 'dis':
        batch_size = config.dis_batch_size
        def collate_fn(batch_data):
            return discriminator_collate_fn(batch_data, tokenizer)
    elif attri == 'gen':
        batch_size = config.gen_big_batch_size

        def collate_fn(batch_data):
            return generator_collate_fn(
                batch_data, tokenizer, config.gen_batch_size, is_train=shuffle)

    elif attri == 'gen_en':
        batch_size = config.gen_en_batch_size

        def collate_fn(batch_data):
            return generator_en_collate_fn(
                batch_data, tokenizer, is_train=shuffle)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader
