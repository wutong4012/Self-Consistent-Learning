import os
import random

import datasets
import torch
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer, BertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data_utlis.sample_sequence import sample_sequence_batch
from data_utlis.sim_gen_dataset import create_dataloader, load_data, set_dataset
from model_utils.sim_gen_model import Generator, Discriminator


class GenSystem(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        print('\nInitialize Generator...')

        self._set_tokenizers_and_models()

    def set_gen_dataset(self):
        self.train_dataset, self.val_dataset = \
            set_dataset(self.config, use_label=True, use_gen=True, 
                        attri='gen', rank=self.global_rank)

    def _set_tokenizers_and_models(self):
        self.gen_tokenizer = T5Tokenizer.from_pretrained(
            self.config.sp_model_path,
            eos_token='<|endoftext|>',
            pad_token='<|endoftext|>',
            extra_ids=0)
        self.gen_tokenizer.add_special_tokens({'bos_token': '<bos>'})
        self.generator = Generator(self.config)

    def train_dataloader(self):
        return create_dataloader(config=self.config, dataset=self.train_dataset,
                                 tokenizer=self.gen_tokenizer, attri='gen', shuffle=True)

    def val_dataloader(self):
        return create_dataloader(config=self.config, dataset=self.val_dataset,
                                 tokenizer=self.gen_tokenizer, attri='gen', shuffle=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.generator.parameters(),
            lr=self.config.learning_rate,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.config.warmup_steps),
            num_training_steps=self.config.gen_train_steps
        )

        # Must be written strictly according to the specification! ! !
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def on_fit_start(self) -> None:
        self.set_gen_dataset()

    def training_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        loss, _ = self.generator.forward(
            batch['total_num'].cuda(),
            batch['prompts_input_ids'].cuda(),
            batch['lengths_input_ids'].cuda(),
            batch['prompts_attention_mask'].cuda(),
        )
        self.log('gen_train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        loss, _ = self.generator.forward(
            batch['total_num'].cuda(),
            batch['prompts_input_ids'].cuda(),
            batch['lengths_input_ids'].cuda(),
            batch['prompts_attention_mask'].cuda()
        )
        self.log('gen_val_loss', loss.item())
        self.log('gen_val_ppl', torch.exp(loss).item())
        return loss

    def generate_samples(self):
        new_data_path = self.config.gen_data_path + \
            f'_cycle_{self.config.cycle + 1}'
        if self.global_rank == 0:
            print('Staring Generating...')
            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path)
        wudao_data = load_data(self.config, self.global_rank, is_wudao=True)

        def _generate_sim_sentence(example):
            torch.cuda.empty_cache()
            input_ids, length_list = [], []
            for item in example['sentence_list']:
                if item is None or item == [] or len(item) <= 10:
                    continue

                # 每段话只随机选一条句子
                random_num = random.sample(range(len(item)), 1)[0]
                cur_input_ids = self.gen_tokenizer(
                    '<bos>“' + item[random_num] + '”的相似句是“', return_tensors='pt'
                ).input_ids.squeeze()[:-1]  # 不能加<eos>

                # 每个样本复制几份
                length = [cur_input_ids.size(0)] * self.config.gen_repeat_times
                cur_input_ids = [cur_input_ids] * self.config.gen_repeat_times

                length_list.extend(length)
                input_ids.extend(cur_input_ids)

            input_ids = pad_sequence(
                [x for x in input_ids], batch_first=True, padding_value=50000)
            length_tensor = torch.tensor(length_list)

            # if self.config.cycle < self.config.gen_anti_cyle:
            top_k, top_p = 0, 0.95
            # else:
            #     top_k, top_p = 500, 0.9
            self.generator.gen.cuda().eval()
            output_ids_list, ppl_list = sample_sequence_batch(
                model=self.generator.gen, context_tokens_tensor=input_ids.cuda(),
                context_length_tensor=length_tensor, repetition_penalty=1.5, max_out_seq=200,
                end_token_id=50000, temperature=1.0, top_k=top_k, top_p=top_p,
            )
            sim_sentence = self.gen_tokenizer.batch_decode(
                output_ids_list, skip_special_tokens=True)
            
            raw_text, sim_text, real_ppl_list = [], [], []
            for idx, item in enumerate(sim_sentence):
                if item.count('”的相似句是“') != 1 or (
                    item.count('“') % 2 != 0 or item.count('”') % 2 != 0):
                    continue

                item = item.replace(' ', '').split('”的相似句是“')
                raw_text.append(item[0][1:])
                sim_text.append(item[1][:-1])
                real_ppl_list.append(-ppl_list[idx])  # 加上负号，使其越大越好
            
            
            return {'text1': raw_text, 'text2': sim_text, '-ppl': real_ppl_list}
        
        if self.global_rank > 0:
            print(f'Rank {self.global_rank} waiting for main process to perform the mapping')
            torch.distributed.barrier()
        
        gen_sim_ds = wudao_data.map(
            _generate_sim_sentence,
            batched=True,
            batch_size=256,
            num_proc=1,
            cache_file_name=new_data_path + '/raw_cache',
            remove_columns=['sentence_list'])
        self.generator.gen.to('cpu')

        feats = datasets.Features({"text1": datasets.Value('string'),
                                   "text2": datasets.Value('string'),
                                   "score": datasets.Value('int8')})
        dis_tokenizer = BertTokenizer.from_pretrained(self.config.discriminator)
        discriminator = Discriminator(self.config)
        def _pre_score(example):
            torch.cuda.empty_cache()
            input_texts = []
            for idx in range(len(example['text1'])):
                input_texts.append(example['text1'][idx] + '[SEP]' + example['text2'][idx])
            input_ids = dis_tokenizer(
                input_texts, padding=True, return_tensors='pt').input_ids
            with torch.no_grad():
                discriminator.to('cuda').eval()
                logits = discriminator.forward(
                    dis_input_ids=input_ids.cuda(), labels=None)
            
            assert logits.size(0) == len(example['-ppl'])
            unite_logits = []
            for idx in range(logits.size(0)):
                unite_logits.append(logits[idx][1].item() * example['-ppl'][idx])
            unite_logits = torch.softmax(torch.tensor(unite_logits), dim=0)
            
            scores = []
            for idx in range(unite_logits.size(0)):
                if unite_logits[idx] >= 0.7:
                    scores.append(1)
                else:
                    scores.append(0)
            print(f'There are {scores.count(1)} Samples to be selected!')
            
            return {'score': scores}
    
        gen_sim_ds = wudao_data.map(
            _pre_score,
            batched=True,
            batch_size=1024,
            num_proc=1,
            features=feats,
            cache_file_name=new_data_path + '/main_cache',
            remove_columns=['-ppl'])
        discriminator.to('cpu')

        if self.global_rank == 0 and self.config.cycle != -1:
            torch.distributed.barrier()

        if self.global_rank == 0:
            print(f'Generate Data Samples is {gen_sim_ds.num_rows}')

            gen_sim_ds.save_to_disk(new_data_path)
            print('gen_data: done!!!')
        if self.config.cycle != -1:
            torch.distributed.barrier()
