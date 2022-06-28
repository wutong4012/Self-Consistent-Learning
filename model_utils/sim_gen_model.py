import json

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from model_utils.gpt2_modeling import GPT2Model


class Discriminator(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.dis = BertForSequenceClassification.from_pretrained(
            config.discriminator, num_labels=2)
        
        if config.pretrain_dis:
            return

        if config.cycle == 0 or config.cycle == -1:
            pt_path = config.dis_model_path
        else:
            pt_path = config.ckpt_model_path + \
                f'/discriminator_cycle_{config.cycle}.ckpt/checkpoint/mp_rank_00_model_states.pt'

        new_dict = {}
        state_dict = torch.load(pt_path, map_location='cpu')['module']
        for k, v in state_dict.items():
            if any([i in k for i in ['module.discriminator.dis.']]):
                new_dict[k[len('module.discriminator.dis.'):]] = v
            else:
                continue
        self.dis.load_state_dict(new_dict)
        print(f'Cycle {config.cycle}: The Discriminator Erlangshen Load Successfully !\n')

    def forward(self, dis_input_ids, labels):
        attention_mask = (dis_input_ids > 0).int()
        # [CLS] -> [bs, hz]
        dis_output = self.dis.forward(
            input_ids=dis_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if labels == None:
            return dis_output.logits

        return dis_output.loss, dis_output.logits


class Generator(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        with open(config.txl_config_path, 'r') as f:
            txl_config = json.load(f)
        self.gen = GPT2Model(
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
        
        if config.cycle == 0 or config.cycle == -1:
            pt_path = config.txl_model_path
        else:
            pt_path = config.ckpt_model_path +\
                f'/generator_cycle_{config.cycle}.ckpt/checkpoint/mp_rank_00_model_states.pt'
        
        new_dict = {}
        state_dict = torch.load(pt_path, map_location='cpu')['module']
        for k, v in state_dict.items():
            if any([i in k for i in ['module.generator.gen.']]):
                new_dict[k[len('module.generator.gen.'):]] = v
            else:
                continue
        if new_dict == {}:
            new_dict = state_dict
        self.gen.load_state_dict(new_dict)
        print(f'Cycle {config.cycle}: The Generator Transformer-XL Load Successfully !\n')

    def forward(self, total_num, gen_input_ids, lengths_input_ids, memory_attention_mask):
        gen_output = self.gen.forward(
            input_ids=gen_input_ids,
            position_ids=None,
            attention_mask=memory_attention_mask,
        )[0]

        shift_logits = gen_output[..., :-1, :].contiguous()  # [bs, seq_len-1, vocab_size]
        shift_labels = gen_input_ids[..., 1:].contiguous()  # [bs, seq_len-1]

        loss_fct = nn.CrossEntropyLoss(reduce=False)  
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)) # [bs*(seq_len-1), ]
        
        lengths_input_ids = lengths_input_ids[..., 1:].contiguous()
        loss = (loss * (
            (lengths_input_ids.view(-1) > 1).int()) / lengths_input_ids.view(-1)
        ).sum() / total_num 
        
        return loss, gen_output
