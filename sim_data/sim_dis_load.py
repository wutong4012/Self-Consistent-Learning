import os
import random
import warnings

import datasets
import torch
from transformers import BertTokenizer

from model_utils.sim_gan_model import Discriminator
from data_utlis.noisy_input_ids import noisy

_BATCH_SIZE = 1024  # bert-2048 / els-1536
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
            '/cognitive_comp/wutong/source/model_base/dis_f1=0.9998.ckpt')['state_dict']
        new_dict = {key[len('discriminator.'):]: val for key,
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
        scores = []
        text1_ids = dis_tokenizer(
            example['text1'], padding=True, return_tensors='pt').input_ids
        noisy_text1_ids = noisy(x=text1_ids, drop_prob=0.05, sub_prob=0.05, shuffle_dist=0,
                                bos_token=101, pad_token=102, vocab_size=21128)
        nosiy_text1 = dis_tokenizer.batch_decode(
            noisy_text1_ids, skip_special_tokens=True)
        input_texts = []
        for idx in range(len(nosiy_text1)):
            input_texts.append(
                nosiy_text1[idx].replace(' ', '') + '[SEP]' + example['text2'][idx])

        input_ids = dis_tokenizer(
            input_texts, padding=True, return_tensors='pt').input_ids
        with torch.no_grad():
            logits = discriminator.forward(
                dis_input_ids=input_ids.cuda(), labels=None)
        scores = torch.argmax(logits, dim=1).tolist()

        return {'score': scores}

    gen_sim_ds = sim_data.map(
        _generate_sim_sentence,
        batched=True,
        batch_size=_BATCH_SIZE,
        keep_in_memory=True,
        num_proc=num_proc,
        remove_columns=['score'])
    print(gen_sim_ds)

    gen_sim_ds.save_to_disk(_CACHE_DATA_PATH)
    print('done')

if __name__ == '__main__':
    if not os.path.exists(_CACHE_DATA_PATH):
        os.makedirs(_CACHE_DATA_PATH)
    generate_arrow_cache()
