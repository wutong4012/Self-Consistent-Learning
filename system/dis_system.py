import torch
from sklearn.metrics import accuracy_score, f1_score
from pytorch_lightning import LightningModule
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data_utlis.predict_dataset import create_predict_dataloader
from data_utlis.sim_gen_dataset import (create_dataloader, set_dataset)
from model_utils.sim_gen_model import Discriminator


class DisSystem(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        print('\nInitialize Discriminator...')

        self._set_tokenizers_and_models()

    def set_dis_dataset(self):
        self.train_dataset, self.val_dataset = \
            set_dataset(self.config, use_label=True, use_gen=True,
                        attri='dis', rank=self.global_rank)

    def _set_tokenizers_and_models(self):
        self.dis_tokenizer = BertTokenizer.from_pretrained(
            self.config.discriminator)
        self.discriminator = Discriminator(self.config)

    def train_dataloader(self):
        if self.global_rank == 0:
            print('**********Start to Prepare the Train Dataloader**********')
        return create_dataloader(config=self.config, dataset=self.train_dataset,
                                 tokenizer=self.dis_tokenizer, attri='dis', shuffle=True)

    def val_dataloader(self):
        if self.global_rank == 0:
            print('**********Start to Prepare the Validation Dataloader**********')
        return create_dataloader(config=self.config, dataset=self.val_dataset,
                                 tokenizer=self.dis_tokenizer, attri='dis', shuffle=False)
    
    def predict_dataloader(self):
        if self.global_rank == 0:
            print('**********Start to Prepare the Predict Dataloader**********')
        return create_predict_dataloader(config=self.config, tokenizer=self.dis_tokenizer,
                                         rank=self.global_rank, attri='dis')

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

    def on_fit_start(self) -> None:
        self.set_dis_dataset()

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

        predictions = torch.argmax(logits, dim=1).tolist()
        if self.config.val_metric == 'f1':
            score = f1_score(batch['labels'].tolist(), predictions)
        elif self.config.val_metric == 'acc':
            score = accuracy_score(batch['labels'].tolist(), predictions)
        self.log('dis_score', score)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()

        with torch.no_grad():
            self.discriminator.to('cuda').eval()
            logits = self.discriminator.forward(
                dis_input_ids=batch['input_ids'].cuda(), labels=None)
            logits = torch.softmax(logits, dim=1)

        return {
            'text1': batch['text1'],
            'text2': batch['text2'],
            'logits': logits,
        }
