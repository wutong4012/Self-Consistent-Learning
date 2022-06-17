import json

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

from model_utils.gpt2_modeling import GPT2Model


class Discriminator(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(config.dis_hidden_size, 2, bias=False)
        self.dis = BertForSequenceClassification.from_pretrained(
            config.discriminator, num_labels=2)

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

        # logits = self.sigmoid(self.linear(dis_output.pooler_output))
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(logits.view(-1), labels.view(-1))

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
        
        if config.cycle == 0:
            self.gen.load_state_dict(torch.load(config.txl_model_path, 
                                                map_location='cpu')['module'])
        else:
            state_dict = torch.load(config.ckpt_model_path +\
                f'/generator_cycle_{config.cycle}.ckpt/checkpoint/mp_rank_00_model_states.pt', 
                map_location='cpu')['module']
            new_dict = {key[len('module.generator.gen.'):]: val for key,
                        val in state_dict.items()}
            self.gen.load_state_dict(new_dict)
        print(f'Cycle {config.cycle}: The Generator Transformer-XL Load Successfully !\n')

    def forward(self, gen_input_ids, lengths_input_ids, memory_attention_mask):
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
            (lengths_input_ids.view(-1) > 1).int()) / lengths_input_ids.view(-1)).sum() 
        
        return loss, gen_output
