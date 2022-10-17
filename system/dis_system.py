import torch
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
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

    def on_fit_start(self) -> None:
        self.set_dis_dataset()

    def set_dis_dataset(self):
        self.train_dataset, self.val_dataset = \
            set_dataset(self.config, attri='dis', dis_tokenizer=self.dis_tokenizer)

    def _set_tokenizers_and_models(self):
        self.dis_tokenizer = AutoTokenizer.from_pretrained(self.config.dis_model_path)
        self.discriminator = Discriminator(self.config, self.dis_tokenizer)

    def train_dataloader(self):
        print('**********Start to Prepare the Train Dataloader**********')
        return create_dataloader(config=self.config, dataset=self.train_dataset,
                                 tokenizer=self.dis_tokenizer, attri='dis', shuffle=True)

    def val_dataloader(self):
        print('**********Start to Prepare the Validation Dataloader**********')
        return create_dataloader(config=self.config, dataset=self.val_dataset,
                                 tokenizer=self.dis_tokenizer, attri='dis', shuffle=False)
    
    def predict_dataloader(self):
        print('**********Start to Prepare the Predict Dataloader**********')
        return create_predict_dataloader(config=self.config, tokenizer=self.dis_tokenizer, attri='dis')

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
        ret = self.discriminator.forward(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                token_type_ids=batch['token_type_ids'].cuda(),
                position_ids=batch['position_ids'].cuda(),
                mlmlabels=batch['mlmlabels'].cuda(),
                clslabels=batch['clslabels'].cuda(),
                clslabels_mask=batch['clslabels_mask'].cuda(),
                mlmlabels_mask=batch['mlmlabels_mask'].cuda(),
            )
        self.log('dis_train_loss', ret['loss_total'], on_step=True, on_epoch=True)

        return ret['loss_total']

    def validation_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        ret = self.discriminator.forward(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                token_type_ids=batch['token_type_ids'].cuda(),
                position_ids=batch['position_ids'].cuda(),
                mlmlabels=batch['mlmlabels'].cuda(),
                clslabels=batch['clslabels'].cuda(),
                clslabels_mask=batch['clslabels_mask'].cuda(),
                mlmlabels_mask=batch['mlmlabels_mask'].cuda(),
            )
        self.log('dis_val_loss', ret['loss_total'])

        return ret['loss_total']

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()

        with torch.no_grad():
            self.discriminator.to('cuda').eval()
            prob = self.discriminator.forward(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                token_type_ids=batch['token_type_ids'].cuda(),
                position_ids=batch['position_ids'].cuda(),
                clslabels_mask=batch['clslabels_mask'].cuda(),
                bt_label_idx=batch['label_idx'].cuda()
            )

        return {
            'sentence1': batch['sentence1'],
            'sentence2': batch['sentence2'],
            'prob': prob,
        }
