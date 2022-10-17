import torch
from sklearn.metrics import f1_score
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AlbertTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule

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
            set_dataset(self.config, use_label=self.config.dis_use_label,
                        use_gen=True, attri='dis', rank=self.global_rank, 
                        dis_tokenizer=self.dis_tokenizer)

    def _set_tokenizers_and_models(self):
        if self.config.chinese:
            self.dis_tokenizer = AutoTokenizer.from_pretrained(
                '/cognitive_comp/wutong/source/model_base/pretrained_zh/' + self.config.bustm_model)
        
        else:
            if self.config.discriminator_en == 'albert_xxlarge':
                self.dis_tokenizer = AlbertTokenizer.from_pretrained(
                    self.config.pretrained_en + self.config.discriminator_en)
            else:
                self.dis_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.pretrained_en + self.config.discriminator_en)
        
        self.discriminator = Discriminator(self.config, self.dis_tokenizer)

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
        # scheduler = get_constant_schedule(optimizer=optimizer)

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
        
        # predictions = torch.argmax(ret['cls_logits', dim=1)
        # f1 = f1_score(batch['labels'].cpu(), predictions.cpu())
        # self.log('dis_f1_score', f1)

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
                # bt_label_idx=batch['label_idx'].cuda()
            )

        return {
            'sentence1': batch['sentence1'],
            'sentence2': batch['sentence2'],
            'prob': prob,
        }
