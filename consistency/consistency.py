import torch, json, datasets
import numpy as np

from tqdm import tqdm
from scipy.stats import wasserstein_distance, entropy

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    T5Tokenizer, AutoTokenizer, AlbertTokenizer, GPT2Tokenizer)

import sys
sys.path.append('/cognitive_comp/wutong/similarity_generation/')
from data_utlis.sim_gen_dataset import SimGanDataset
from data_utlis.sample_sequence import (
    sample_sequence_batch, sample_sequence_batch_en)
from model_utils.sim_gen_model import (
    Generator, Discriminator, Generator_EN)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


# hyper parameters
all_cycle = 9
data_name = 'mrpc'
ckpt_name = 'mrpc0'
chinese = 0


def gen_pred_collate(batch_data, gen_tokenizer):
    input_ids, length_list = [], []
    for item in batch_data:
        if chinese:
            cur_input_ids = gen_tokenizer(
                '<bos>“' + item['text1'] + '”的相似句是“', return_tensors='pt'
            ).input_ids.squeeze()[:-1]  # 不能加<eos>
        else:
            cur_input_ids = gen_tokenizer(
                '"' + item['text1'] + '" is similar to "', return_tensors='pt'
            ).input_ids.squeeze()[1:]  # 去掉<bos>

        # 每个样本复制 N 份
        length = [cur_input_ids.size(0)] * 1
        cur_input_ids = [cur_input_ids] * 1

        length_list.extend(length)
        input_ids.extend(cur_input_ids)

    if config.chinese:
        input_ids = pad_sequence(
            [x for x in input_ids], batch_first=True, 
            padding_value=gen_tokenizer.pad_token_id)
    else:
        input_ids = pad_sequence(
            [x for x in input_ids], batch_first=True, 
            padding_value=gen_tokenizer.pad_token_id)
    length_tensor = torch.tensor(length_list)

    return {
        'input_ids': input_ids,
        'length_tensor': length_tensor,
    }


def discriminator_collate_fn(batch_data, tokenizer):
    dis_text_input_ids, ppl_list = [], []
    for item in batch_data:
        dis_text = item['text1'] + '[SEP]' + item['text2']
        input_ids = tokenizer(dis_text, return_tensors='pt').input_ids.squeeze()
        dis_text_input_ids.append(input_ids)
        ppl_list.append(item['ppl'])

    dis_text_input_ids = pad_sequence([x for x in dis_text_input_ids],
                                      batch_first=True, 
                                      padding_value=tokenizer.pad_token_id)

    return {
        'dis_text_input_ids': dis_text_input_ids,
        'ppl': ppl_list
    }


class Config:
    cycle = 0
    zero_shot = 1
    chinese = chinese
    data_name = data_name
    warm_up_model = True
    pretrain_dis = False
    opt_name = 'opt-2.7b'
    discriminator_en = 'albert_xxlarge'
    discriminator_zh = 'roberta_large'
    pretrained_en = '/cognitive_comp/wutong/source/model_base/pretrained_en/'
    pretrained_zh = '/cognitive_comp/wutong/source/model_base/pretrained_zh/'
    
    opt_model_path = '/cognitive_comp/wutong/source/model_base/model_en/'
    txl_config_path = '/cognitive_comp/wutong/similarity_generation/model_utils/txl_5B_config.json'
    txl_model_path = '/cognitive_comp/wutong/source/model_base/model_zh/txl_zh_5.0B.pt'
    ckpt_model_path = '/cognitive_comp/wutong/similarity_generation/all_checkpoints/' + ckpt_name
    # ckpt_model_path = '/cognitive_comp/wutong/similarity_generation/experiments/lightning_logs/checkpoints/mrpc_test'
config = Config()


if chinese:
    gen_tokenizer = T5Tokenizer.from_pretrained(
        '/cognitive_comp/wutong/source/model_base/chinese_sentencepiece/cog-pretrain.model',
        eos_token='<|endoftext|>',
        pad_token='<|endoftext|>',
        extra_ids=0)
    gen_tokenizer.add_special_tokens({'bos_token': '<bos>'})
    dis_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_zh + config.discriminator_zh)
    
else:
    gen_tokenizer = GPT2Tokenizer.from_pretrained(config.opt_model_path + 'opt-2.7b')
    dis_tokenizer = AlbertTokenizer.from_pretrained(config.pretrained_en + config.discriminator_en)


all_w, all_e, all_kl, all_h = [], [], [], []
for idx in range(all_cycle):
    print(f'Cycle {idx}...')
    config.cycle = idx
    data = datasets.load_from_disk('/cognitive_comp/wutong/source/sim_data/sim_test_data/' + data_name)
    
    # data
    predict_dataset = SimGanDataset(data)
    def collate_fn(batch_data):
        return gen_pred_collate(batch_data, gen_tokenizer)
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=300,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # model
    if chinese:
        generator = Generator(config)
    else:
        generator = Generator_EN(config)
    generator.half().eval().cuda()
    
    # inference
    ppl_list, sim_sent_list = [], []
    for batch in dataloader:
        torch.cuda.empty_cache()
        
        if chinese:
            output_dict = sample_sequence_batch(
                model=generator.gen, context_tokens_tensor=batch['input_ids'].cuda(),
                context_length_tensor=batch['length_tensor'], repetition_penalty=1.0, max_out_seq=200,
                end_token_id=gen_tokenizer.eos_token_id, temperature=1.0, top_k=1, top_p=0.0,
            )
            ppl_list.extend(output_dict['ppl_list'])  # ppl_list / prob_list

        else:
            output_dict = sample_sequence_batch_en(
                model=generator.gen, context_tokens_tensor=batch['input_ids'].cuda(),
                context_length_tensor=batch['length_tensor'], repetition_penalty=1.0, max_out_seq=100,
                end_token_id=gen_tokenizer.eos_token_id, temperature=1.0, top_k=1, top_p=0.0,
            )
            ppl_list.extend(output_dict['ppl_list'])  # ppl_list / prob_list
        
        sim_sent_list.extend(gen_tokenizer.batch_decode(
            output_dict['ids_list'], skip_special_tokens=True))

    # save data
    with open('/cognitive_comp/wutong/similarity_generation/ipynb/data.json', 'w') as wp:
        if chinese:
            for jdx, item in tqdm(enumerate(sim_sent_list)):
                item = item.replace(' ', '').split('”的相似句是“')
                if len(item) == 2:
                    wp.write(json.dumps({'text1': item[0][1:],
                                         'text2': item[1][:-1],
                                         'ppl': ppl_list[jdx]}, ensure_ascii=False) + '\n')
            wp.close()
        else:
            for jdx, item in tqdm(enumerate(sim_sent_list)):
                item = item.replace('\n', '').split('\" is similar to \"')
                if len(item) == 2:
                    wp.write(json.dumps({'text1': item[0][1:],
                                         'text2': item[1].split('"')[0],
                                         'ppl': ppl_list[jdx]}, ensure_ascii=False) + '\n')
            wp.close()

    path = '/cognitive_comp/wutong/similarity_generation/ipynb/data.json'
    feats = datasets.Features({"text1": datasets.Value('string'), 
                               "text2": datasets.Value('string'),
                               "ppl": datasets.Value('float64'),
                            })
    ds = (datasets.load_dataset('json', data_files=path, 
                                cache_dir='/cognitive_comp/wutong/source/data_base/huggingface-cache',
                                features=feats)['train'])
    ds.save_to_disk('/cognitive_comp/wutong/similarity_generation/ipynb/data')

    # data
    data = datasets.load_from_disk('/cognitive_comp/wutong/similarity_generation/ipynb/data')
    dataset = SimGanDataset(data)
    def collate_fn(batch_data):
        return discriminator_collate_fn(batch_data, dis_tokenizer)
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=500,  # afqmc-300
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    # model
    discriminator = Discriminator(config)
    discriminator.cuda().eval()
    
    # inference
    with torch.no_grad():
        pred_list, ppl_list = [], []
        for batch in dataloader:
            torch.cuda.empty_cache()
            logits = discriminator.forward(
                batch['dis_text_input_ids'].cuda(),
                None
            )
            ppl_list.extend(batch['ppl'])
            
            pred = torch.softmax(logits, dim=1)
            for jdx in range(pred.size(0)):
                pred_list.append(pred[jdx][1].item())
    
    # save data    
    def add_prob(example, idx):
        return {'prob': pred_list[idx]}
    data = data.map(add_prob, with_indices=True)
    data.save_to_disk(f'/cognitive_comp/wutong/similarity_generation/ipynb/data_cycle_{idx}')
    print(f'Finished Cycle {idx}!')
    
    # cal dis
    e_d = np.sqrt(np.sum(np.square(np.array(data['ppl']) - np.array(data['prob']))))
    all_e.append(e_d)

    kl = entropy(data['prob'], data['ppl'])
    all_kl.append(kl)

    w_d = wasserstein_distance(data['ppl'], data['prob'])
    all_w.append(w_d)

    h_d = 1 / np.sqrt(2) * np.linalg.norm(np.sqrt(data['ppl']) - np.sqrt(data['prob']))
    all_h.append(h_d)
    
# print(all_e)
print(all_kl)
# print(all_w)
# print(all_h)
