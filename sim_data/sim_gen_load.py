import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import datasets
import torch
from transformers import T5Tokenizer
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append('/cognitive_comp/wutong/similarity_generation/')
from model_utils.gpt2_for_inference import GPT2Model
from data_utlis.sample_sequence import sample_sequence_batch


_NUM_FILES = 1
_WUDAO_DATA_PATH = '/cognitive_comp/wutong/source/data_base/wudao_sentences/'
_CACHE_DATA_PATH = '/cognitive_comp/wutong/source/data_base/gen_sim_data_cycle_0'


def load_wudao_data():
    cache_dict_paths = glob.glob(os.path.join(_WUDAO_DATA_PATH, '*'))
    wudao_ds, res = [], []
    p = ProcessPoolExecutor(max_workers=1)

    for path in cache_dict_paths[:_NUM_FILES]:
        res.append(p.submit(datasets.load_from_disk, path))
    p.shutdown(wait=True)
    for future in res:
        wudao_ds.append(future.result())
    wudao_ds = datasets.concatenate_datasets(wudao_ds)

    print(f'There ara total {wudao_ds.num_rows} sentences !')
    random_list = random.sample(range(wudao_ds.num_rows), 3)
    for i in random_list:
        print("Examples: {}".format(wudao_ds['sentence_list'][i]))

    return wudao_ds


def load_txl_model():
    with open('/cognitive_comp/wutong/similarity_generation/model_utils/txl_5B_config.json', 'r') as f:
        txl_config = json.load(f)
    generator = GPT2Model(
        num_layers=txl_config['num_layers'],
        vocab_size=txl_config['vocab_size'],
        hidden_size=txl_config['hidden_size'],
        num_attention_heads=txl_config['num_attention_heads'],
        embedding_dropout_prob=txl_config['embedding_dropout_prob'],
        attention_dropout_prob=txl_config['attention_dropout_prob'],
        output_dropout_prob=txl_config['output_dropout_prob'],
        max_sequence_length=txl_config['max_sequence_length'],
        max_memory_length=txl_config['max_memory_length'],
        checkpoint_activations=txl_config['checkpoint_activations'],
        checkpoint_num_layers=txl_config['checkpoint_num_layers'],
        parallel_output=txl_config['parallel_output'],
        relative_encoding=txl_config['relative_encoding']
    )
    generator.load_state_dict(torch.load(
        '/cognitive_comp/wutong/source/model_base/txl_zh_5.0B.pt')['module'])

    return generator


def generate_arrow_cache(num_proc=1) -> None:
    wudao_ds = load_wudao_data()

    gen_tokenizer = T5Tokenizer.from_pretrained(
        '/cognitive_comp/wutong/source/model_base/chinese_sentencepiece/cog-pretrain.model',
        eos_token='<|endoftext|>',
        pad_token='<|endoftext|>',
        extra_ids=0)
    gen_tokenizer.add_special_tokens({'bos_token': '<bos>'})
    generator = load_txl_model().to('cuda')
    print('Load model and tokenizer successfully!')

    def _generate_sim_sentence(example):
        torch.cuda.empty_cache()
        sim_text = []
        input_ids, length_list = [], []
        for item in example['sentence_list']:
            if item is None or item == [] or len(item) <= 10:
                continue

            # 每段话只随机选一条句子
            random.seed(42)
            random_num = random.sample(range(len(item)), 1)[0]
            cur_input_ids = gen_tokenizer(
                '<bos>“' + item[random_num] + '”的相似句是“', return_tensors='pt').input_ids.squeeze()[:-1]  # 不能加<eos>

            length = [cur_input_ids.size(0)] * 5
            cur_input_ids = [cur_input_ids] * 5
            length_list.extend(length)
            input_ids.extend(cur_input_ids)

        input_ids = pad_sequence(
            [x for x in input_ids], batch_first=True, padding_value=50000)
        length_tensor = torch.tensor(length_list)

        output_ids_list = sample_sequence_batch(
            model=generator.cuda(), context_tokens_tensor=input_ids.cuda(), context_length_tensor=length_tensor,
            repetition_penalty=1.5, max_out_seq=200, end_token_id=50000, temperature=1.5, top_k=0, top_p=0.82,
        )
        sim_sentence = gen_tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)

        raw_text, sim_text = [], []
        for item in sim_sentence:
            if item.count('”的相似句是“') != 1 or (
                item.count('“') % 2 != 0 or item.count('”') % 2 != 0
            ) or len(item) <= 10:
                continue

            item = item.replace(' ', '').split('”的相似句是“')
            raw_text.append(item[0][1:])
            sim_text.append(item[1][:-1])
        score = [1] * len(raw_text)

        return {'text1': raw_text, 'text2': sim_text, 'score': score}

    feats = datasets.Features({"text1": datasets.Value('string'), 
                               "text2": datasets.Value('string'), 
                               "score": datasets.Value('int8')})
    gen_sim_ds = wudao_ds.map(
        _generate_sim_sentence,
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        features=feats,
        remove_columns=['sentence_list'])
    print(gen_sim_ds)

    gen_sim_ds.save_to_disk(_CACHE_DATA_PATH)
    print('done')


if __name__ == '__main__':
    if not os.path.exists(_CACHE_DATA_PATH):
        os.makedirs(_CACHE_DATA_PATH)
    generate_arrow_cache()
