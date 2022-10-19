import sys
import torch

from time import time
from flask import Flask, jsonify, request
from transformers import T5Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.append('/cognitive_comp/wutong/similarity_generation/')
from data_utlis.sample_sequence import sample_sequence_batch
from model_utils.sim_gen_model import Generator


import socket
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
print(ip)


def gen_pred_collate(batch_data, gen_tokenizer):
    input_ids, length_list = [], []
    for item in batch_data:
        cur_input_ids = gen_tokenizer(
            '<bos>“' + item + '”的相似句是“', return_tensors='pt'
        ).input_ids.squeeze()[:-1]  # 不能加<eos>

        length_list.extend([cur_input_ids.size(0)])
        input_ids.extend([cur_input_ids])

    input_ids = pad_sequence(
        [x for x in input_ids], batch_first=True, 
        padding_value=gen_tokenizer.pad_token_id)
    length_tensor = torch.tensor(length_list)

    return {
        'input_ids': input_ids,
        'length_tensor': length_tensor,
    }
    

class SimGenDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Config:
    cycle = 10
    chinese = 1
    ckpt_name = 'qqp0'
    txl_config_path = '/cognitive_comp/wutong/similarity_generation/model_utils/txl_5B_config.json'
    txl_model_path = '/cognitive_comp/wutong/source/model_base/model_zh/txl_zh_5.0B.pt'
    ckpt_model_path = '/cognitive_comp/wutong/similarity_generation/all_checkpoints/' + ckpt_name
config = Config()

generator = Generator(config)
generator.half().eval().cuda()

gen_tokenizer = T5Tokenizer.from_pretrained(
    '/cognitive_comp/wutong/source/model_base/chinese_sentencepiece/cog-pretrain.model',
    eos_token='<|endoftext|>',
    pad_token='<|endoftext|>',
    extra_ids=0)
gen_tokenizer.add_special_tokens({'bos_token': '<bos>'})


def collate_fn(batch_data):
    return gen_pred_collate(batch_data, gen_tokenizer)


app = Flask(__name__)

@app.route("/simgen", methods=["POST"])
def text_simulate():
    request_json = request.get_json()
    batch_size = request_json.get("batch_size", 256)
    top_k = request_json.get("top_k", 0)
    top_p = request_json.get("top_p", 0.9)
    temperature = request_json.get("temperature", 1.0)
    repetition_penalty = request_json.get("repetition_penalty", 1.0)
    max_out_length = request_json.get("max_out_length", 128)
    sent_inputs = request_json.get("sent_inputs", None)
    
    predict_dataset = SimGenDataset(sent_inputs)
    dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    sim_sent_list = []
    start = time()
    for batch in dataloader:
        output_dict = sample_sequence_batch(
                model=generator.gen, context_tokens_tensor=batch['input_ids'].cuda(),
                context_length_tensor=batch['length_tensor'], repetition_penalty=repetition_penalty, 
                max_out_seq=max_out_length, end_token_id=50000, temperature=temperature, top_k=top_k, top_p=top_p,
            )
        sim_sent_list.extend(
            gen_tokenizer.batch_decode(output_dict['ids_list'], skip_special_tokens=True))
    used_time = time() - start
    
    raw_text, sim_text = [], []
    for item in sim_sent_list:

        item = item.replace(' ', '').split('”的相似句是“')
        if item[0][1:] and item[1][:-1]:
            raw_text.append(item[0][1:])
            sim_text.append(item[1][:-1])

    return jsonify(
        time = used_time,
        origin_sentence = raw_text,
        generated_sentence = sim_text,
    )


ip='192.168.52.175'
app.run(host=ip, port=42345)
