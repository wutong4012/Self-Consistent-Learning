import json

import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM, OPTForCausalLM)

from model_utils.gpt2_modeling import GPT2Model


class Discriminator(nn.Module):

    def __init__(self, config, tokenizer) -> None:
        super().__init__()
        pt_path = '/cognitive_comp/wutong/source/model_base/pretrained_zh/' + config.bustm_model
        self.dis = AutoModelForMaskedLM.from_pretrained(pt_path) 

        self.yes_token = tokenizer.encode("是")[1]
        self.no_token = tokenizer.encode("非")[1]

        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
        self.KL_criterion = torch.nn.KLDivLoss(reduction='batchmean')

        self.temperature = 1
        self.do_annealing = None

        if config.warm_up_model:
            print('Use Warm Up Model...')
            if config.cycle == 0 or config.cycle == -1:
                pt_path = '/cognitive_comp/wutong/' + config.model_version
                state_dict = torch.load(pt_path, map_location='cpu')
                new_dict = {}
                for k, v in state_dict.items():
                    if any([i in k for i in ['bert_encoder.']]):
                        new_dict[k[len('bert_encoder.'):]] = v
                    else:
                        continue
                self.dis.load_state_dict(new_dict)
                print(f'The warm up model path is {pt_path}!')
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
                pt_path = '/cognitive_comp/wutong/source/model_base/pretrained_zh/' + config.bustm_model  ## 
                self.dis = AutoModelForMaskedLM.from_pretrained(pt_path)
                self.dis.load_state_dict(new_dict)
        
        print(f'Cycle {config.cycle}: The Discriminator Load Successfully !\n')

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids=None,
                mlmlabels=None, 
                clslabels=None, 
                clslabels_mask=None, 
                mlmlabels_mask=None,
                bt_label_idx=None,
                t_probs=None,
                t_logits=None,
                do_soft_label=False,
                annealing_coeff=0,
                do_cc_loss=False,
                ):
                
        batch_size,seq_len=input_ids.shape
        outputs = self.dis(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            labels=mlmlabels)  # (bsz, seq, dim)
        mask_loss = outputs.loss
        mlm_logits = outputs.logits
        cls_logits = mlm_logits[:,:,self.yes_token].view(-1,seq_len)+clslabels_mask
        
        loss_ce = 0.0

        if mlmlabels == None:
            probs = torch.nn.functional.softmax(cls_logits, dim=-1)
            # label_idx = torch.stack([sample[:-1] for sample in bt_label_idx], dim=0)
            # probs_ = torch.gather(probs, dim=1, index=label_idx) / self.temperature
            return probs
        else:
            cls_loss = self.loss_func(cls_logits,clslabels)
            loss_total = mask_loss+cls_loss
            # all_loss = mask_loss
        
        # 计算kl 散度loss
        kl_loss = 0.0
        if do_soft_label:
            probs_ = []
            probs = torch.nn.functional.softmax(cls_logits, dim=-1)
            
            label_idx = torch.stack([sample[:-1] for sample in bt_label_idx], dim=0)
                
            probs_ = torch.gather(probs, dim=1, index=label_idx) / self.temperature
            
            kl_loss = self.KL_criterion(probs_.log(), t_probs/self.temperature) 
            
            if self.do_annealing:
                loss_total = annealing_coeff * cls_loss + (1 - annealing_coeff) * kl_loss + mask_loss
            else:
                loss_total += kl_loss

        ret = {
            'loss_total': loss_total,
            'mlm_logits': mlm_logits,
            'cls_logits': cls_logits,
            'loss_ce': loss_ce,
            'kl_loss': kl_loss
        }
        
        return ret


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
            print('Use Warm Up Model...')
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


class Generator_EN(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        
        self.gen = OPTForCausalLM.from_pretrained(config.opt_model_path + config.opt_name)
        
        if config.cycle == 0 or config.cycle == -1:
            print('Use Warm Up Model...')
            pt_path = config.opt_model_path + 'opt-2.7b.pt'
        
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
        
    def forward(self, input_ids, lengths):
        attention_mask = (input_ids > 1).int()
        gen_output = self.gen.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits
        
        shift_logits = gen_output[..., :-1, :].contiguous()  # [bs, seq_len-1, vocab_size]
        shift_labels = input_ids[..., 1:].contiguous()  # [bs, seq_len-1]

        loss_fct = nn.CrossEntropyLoss(reduce=False)  
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)) # [bs*(seq_len-1), ]
        
        lengths = lengths[..., 1:].contiguous()
        loss = (loss * lengths.view(-1)).sum() / input_ids.size(0) 
        
        return loss, gen_output
