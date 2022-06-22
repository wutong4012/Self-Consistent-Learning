import os

import evaluate
import torch
from pytorch_lightning import LightningModule
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data_utlis.sim_gen_dataset import (create_dataloader, load_data,
                                        set_dataset)
from model_utils.sim_gen_model import Discriminator


class DisSystem(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        print('\nInitialize Discriminator...')

        self.f1_metric = evaluate.load("f1")
        self._set_tokenizers_and_models()

    def set_dis_dataset(self):
        self.train_dataset, self.val_dataset = \
            set_dataset(self.config, use_label=True, use_gen=True, attri='dis')

    def _set_tokenizers_and_models(self):
        self.dis_tokenizer = BertTokenizer.from_pretrained(
            self.config.discriminator)
        self.discriminator = Discriminator(self.config)

    def train_dataloader(self):
        return create_dataloader(config=self.config, dataset=self.train_dataset,
                                 tokenizer=self.dis_tokenizer, attri='dis', shuffle=True)

    def val_dataloader(self):
        return create_dataloader(config=self.config, dataset=self.val_dataset,
                                 tokenizer=self.dis_tokenizer, attri='dis', shuffle=False)

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.discriminator.parameters(),
            lr=self.config.learning_rate,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.config.warmup_steps),
            num_training_steps=self.config.dis_train_steps
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

    def training_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        loss, _ = self.discriminator.forward(
            batch['dis_text_input_ids'].cuda(),
            batch['labels'].cuda(),
        )
        self.log('dis_train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        loss, logits = self.discriminator.forward(
            batch['dis_text_input_ids'].cuda(),
            batch['labels'].cuda()
        )
        self.log('dis_val_loss', loss.item())

        predictions = torch.argmax(logits, dim=1)
        f1_score = self.f1_metric.compute(
            references=batch['labels'],
            predictions=predictions
        )
        self.log('dis_f1_score', f1_score['f1'])

        return loss

    def judge_similarity(self):
        new_data_path = self.config.score_data_path + \
            f'_cycle_{self.config.cycle + 1}'
        if self.global_rank == 0:
            print('Staring Scoring...')
            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path)
        generated_data = load_data(self.config, is_labeled=False,
                                   is_score=True, attri='dis')

        def _generate_sim_sentence(example):
            torch.cuda.empty_cache()
            input_texts = []
            for idx in range(len(example['text1'])):
                input_texts.append(
                    example['text1'][idx] + '[SEP]' + example['text2'][idx])

            input_ids = self.dis_tokenizer(
                input_texts, padding=True, return_tensors='pt').input_ids
            with torch.no_grad():
                self.discriminator.to('cuda').eval()
                logits = self.discriminator.forward(
                    dis_input_ids=input_ids.cuda(), labels=None)
                logits = torch.softmax(logits, dim=1)

            assert len(example['text1']) == logits.size(0)
            for idx, item in enumerate(logits):
                if item[1] >= self.config.dis_threshold:
                    example['score'][idx] = 1
                elif item[0] >= self.config.dis_threshold:
                    example['score'][idx] = 0
                else:
                    example['score'][idx] = -5

            return example

        score_sim_ds = generated_data.map(
            _generate_sim_sentence,
            batched=True,
            batch_size=1280,
            num_proc=1,
            cache_file_name=new_data_path + '/raw_cache')
        score_sim_ds = score_sim_ds.filter(lambda example: example['score'] != -5,
                                           cache_file_name=new_data_path+'/main_cache')
        
        if self.global_rank == 0:
            print(f'Score Data Samples is {score_sim_ds.num_rows}')

            score_sim_ds.save_to_disk(new_data_path)
            print('score_data: done!!!')
        torch.distributed.barrier()
