import os
import time
import evaluate
import torch
from pytorch_lightning import LightningModule
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data_utlis.sim_gan_dataset import (create_dataloader, load_data,
                                        set_dataset)
from data_utlis.noisy_input_ids import noisy
from model_utils.sim_gan_model import Discriminator


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
        self.dis_tokenizer = BertTokenizer.from_pretrained(self.config.discriminator)
        self.discriminator = Discriminator(self.config)
        
        if self.config.cycle == 0:
            state_dict = torch.load(self.config.dis_model_path, 
                                    map_location='cpu')['state_dict']
            new_dict = {key[len('discriminator.'):]: val for key,
                        val in state_dict.items()}
            self.discriminator.load_state_dict(new_dict)
        else:
            state_dict = torch.load(self.config.ckpt_model_path +\
                f'/discriminator_cycle_{self.config.cycle}.ckpt/checkpoint/mp_rank_00_model_states.pt', 
                map_location='cpu')['module']
            new_dict = {key[len('module.discriminator.'):]: val for key,
                        val in state_dict.items()}
            self.discriminator.load_state_dict(new_dict)
        print(f'Cycle {self.config.cycle}: The Discriminator Erlangshen Load Successfully !\n')

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
            num_training_steps=200
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
            references=batch['labels'].squeeze().cuda(),
            predictions=predictions
        )
        self.log('dis_f1_score', f1_score['f1'])
        
        return loss
    
    def judge_similarity(self):
        if self.global_rank == 0:
            print('Staring Scoring...')
        generated_data = load_data(self.config, is_labeled=False, 
                                   is_score=True, attri='dis')
        
        def _generate_sim_sentence(example):
            torch.cuda.empty_cache()
            scores = []
            text1_ids = self.dis_tokenizer(
                example['text1'], padding=True, return_tensors='pt').input_ids
            noisy_text1_ids = noisy(x=text1_ids, drop_prob=0.05, sub_prob=0.05, shuffle_dist=0,
                                    bos_token=101, pad_token=102, vocab_size=21128)
            nosiy_text1 = self.dis_tokenizer.batch_decode(
                noisy_text1_ids, skip_special_tokens=True)
            input_texts = []
            for idx in range(len(nosiy_text1)):
                input_texts.append(
                    nosiy_text1[idx].replace(' ', '') + '[SEP]' + example['text2'][idx])

            input_ids = self.dis_tokenizer(
                input_texts, padding=True, return_tensors='pt').input_ids
            with torch.no_grad():
                self.discriminator.to('cuda').eval()
                logits = self.discriminator.forward(
                    dis_input_ids=input_ids.cuda(), labels=None)
            scores = torch.argmax(logits, dim=1).tolist()

            return {'score': scores}

        score_sim_ds = generated_data.map(
            _generate_sim_sentence,
            batched=True,
            batch_size=1536,
            keep_in_memory=True,
            num_proc=1)
        print(score_sim_ds)

        new_data_path = self.config.score_data_path + f'_cycle_{self.config.cycle + 1}'
        if not os.path.exists(new_data_path):
            os.makedirs(new_data_path)
        if self.global_rank == 0:
            score_sim_ds.save_to_disk(new_data_path)
            print('score_data: done!!!')
        else:
            time.sleep(15)
