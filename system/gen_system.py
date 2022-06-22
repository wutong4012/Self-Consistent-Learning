import os
import random

import datasets
import torch
from pytorch_lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data_utlis.sample_sequence import sample_sequence_batch
from data_utlis.sim_gen_dataset import create_dataloader, load_data, set_dataset
from model_utils.sim_gen_model import Generator


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
            batch['prompts_input_ids'].cuda(),
            batch['lengths_input_ids'].cuda(),
            batch['prompts_attention_mask'].cuda(),
        )
        self.log('gen_train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        loss, _ = self.generator.forward(
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
        wudao_data = load_data(self.config, is_wudao=True)

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

            if self.config.cycle < self.config.gen_anti_cyle:
                top_k, top_p = 0, 0.95
            else:
                top_k, top_p = 200, 0.9
            self.generator.gen.cuda().eval()
            output_ids_list = sample_sequence_batch(
                model=self.generator.gen, context_tokens_tensor=input_ids.cuda(),
                context_length_tensor=length_tensor, repetition_penalty=1.5, max_out_seq=200,
                end_token_id=50000, temperature=1.0, top_k=top_k, top_p=top_p,
            )
            sim_sentence = self.gen_tokenizer.batch_decode(
                output_ids_list, skip_special_tokens=True)

            raw_text, sim_text = [], []
            for item in sim_sentence:
                if item.count('”的相似句是“') != 1 or (
                    item.count('“') % 2 != 0 or item.count('”') % 2 != 0):
                    continue

                item = item.replace(' ', '').split('”的相似句是“')
                raw_text.append(item[0][1:])
                sim_text.append(item[1][:-1])

            if self.config.cycle < self.config.gen_anti_cyle:
                scores = [0] * len(raw_text)
            else:
                scores = [1] * len(raw_text)

            return {'text1': raw_text, 'text2': sim_text, 'score': scores}

        feats = datasets.Features({"text1": datasets.Value('string'),
                                    "text2": datasets.Value('string'),
                                    "score": datasets.Value('int8')})
        gen_sim_ds = wudao_data.map(
            _generate_sim_sentence,
            batched=True,
            batch_size=192,
            num_proc=1,
            features=feats,
            cache_file_name=new_data_path + '/main_cache',
            remove_columns=['sentence_list'])
        self.generator.gen.cpu()

        if self.global_rank == 0:
            print(f'Generate Data Samples is {gen_sim_ds.num_rows}')

            gen_sim_ds.save_to_disk(new_data_path)
            print('gen_data: done!!!')
        torch.distributed.barrier()
