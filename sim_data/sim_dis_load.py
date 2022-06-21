import os
import random
import warnings

import datasets
import torch
from transformers import BertTokenizer

import sys
sys.path.append('/cognitive_comp/wutong/similarity_generation/')
from model_utils.sim_gen_model import Discriminator

_BATCH_SIZE = 1280  # bert-2048 / els-1280
_DISCRIMINATOR = 'erlangshen'
_CACHE_DATA_PATH = '/cognitive_comp/wutong/source/data_base/score_sim_data_cycle_0'
warnings.filterwarnings('ignore')


def load_gen_data():
    sim_data_path = '/cognitive_comp/wutong/source/data_base/gen_sim_data_cycle_0'
    sim_data = datasets.load_from_disk(sim_data_path)
    print(sim_data)

    random_list = random.sample(range(sim_data.num_rows), 10)
    for i in random_list:
        print("Examples: {}".format(sim_data[i]))

    return sim_data


class Config:
    if _DISCRIMINATOR == 'bert':
        dis_hidden_size = 768
        discriminator = 'bert-base-chinese'

    elif _DISCRIMINATOR == 'erlangshen':
        dis_hidden_size = 1024
        discriminator = 'IDEA-CCNL/Erlangshen-Roberta-330M-Similarity'


def load_dis_model():
    config = Config()
    dis_model = Discriminator(config)

    if _DISCRIMINATOR == 'bert':
        state_dict = torch.load(
            '/cognitive_comp/wutong/source/model_base/discriminator_bert.ckpt')['state_dict']
        new_dict = {key[len('discriminator.'):]: val for key,
                    val in state_dict.items()}

    elif _DISCRIMINATOR == 'erlangshen':
        state_dict = torch.load(
            '/cognitive_comp/wutong/source/model_base/discriminator.pt', 
            map_location='cpu')['module']
        new_dict = {key[len('module.discriminator.'):]: val for key,
                    val in state_dict.items()}

    dis_model.load_state_dict(new_dict)

    return dis_model


def generate_arrow_cache(num_proc=1) -> None:
    sim_data = load_gen_data()

    if _DISCRIMINATOR == 'bert':
        dis_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    elif _DISCRIMINATOR == 'erlangshen':
        dis_tokenizer = BertTokenizer.from_pretrained(
            'IDEA-CCNL/Erlangshen-Roberta-330M-Similarity')

    discriminator = load_dis_model().to('cuda')

    def _generate_sim_sentence(example):
        torch.cuda.empty_cache()
        scores, input_texts = [], []
        for idx in range(len(example['text1'])):
            input_texts.append(
                example['text1'][idx] + '[SEP]' + example['text2'][idx])

        input_ids = dis_tokenizer(
            input_texts, padding=True, return_tensors='pt').input_ids
        with torch.no_grad():
            logits = discriminator.forward(
                dis_input_ids=input_ids.cuda(), labels=None)
            logits = torch.softmax(logits, dim=1)

        scores = []
        for item in logits:
            if item[1] >= 0.7:
                scores.append(1)
            else:
                scores.append(0)
    
        return {'score': scores}

    feats = datasets.Features({"text1": datasets.Value('string'), 
                               "text2": datasets.Value('string'), 
                               "score": datasets.Value('int8')})
    gen_sim_ds = sim_data.map(
        _generate_sim_sentence,
        batched=True,
        batch_size=_BATCH_SIZE,
        keep_in_memory=True,
        num_proc=num_proc,
        features=feats,
        remove_columns=['score'])
    print(gen_sim_ds)

    gen_sim_ds.save_to_disk(_CACHE_DATA_PATH)
    print('done')


if __name__ == '__main__':
    if not os.path.exists(_CACHE_DATA_PATH):
        os.makedirs(_CACHE_DATA_PATH)
    generate_arrow_cache()
