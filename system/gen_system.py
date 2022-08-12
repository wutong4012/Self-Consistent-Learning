import torch
from pytorch_lightning import LightningModule
from transformers import T5Tokenizer, GPT2Tokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from data_utlis.predict_dataset import create_predict_dataloader
from data_utlis.sample_sequence import sample_sequence_batch
from data_utlis.sim_gen_dataset import create_dataloader, set_dataset
from model_utils.sim_gen_model import Generator, Generator_EN


class GenSystem(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        print('\nInitialize Generator...')

        self._set_tokenizers_and_models()

    def set_gen_dataset(self):
        if self.config.chinese:
            attri = 'gen'
        else:
            attri = 'gen_en'
        self.train_dataset, self.val_dataset = \
            set_dataset(self.config, use_label=True, 
                        use_gen=True, attri=attri, rank=self.global_rank)

    def _set_tokenizers_and_models(self):
        if self.config.chinese:
            self.gen_tokenizer = T5Tokenizer.from_pretrained(
                self.config.sp_model_path,
                eos_token='<|endoftext|>',
                pad_token='<|endoftext|>',
                extra_ids=0)
            self.generator = Generator(self.config)

        else:
            self.gen_tokenizer = GPT2Tokenizer.from_pretrained(
                '/cognitive_comp/wutong/source/model_base/opt-2.7b')
            self.generator = Generator_EN(self.config)


    def train_dataloader(self):
        if self.global_rank == 0:
            print('**********Start to Prepare the Train Dataloader**********')
        if self.config.chinese:
            attri = 'gen'
        else:
            attri = 'gen_en'
        return create_dataloader(config=self.config, dataset=self.train_dataset,
                                 tokenizer=self.gen_tokenizer, attri=attri, shuffle=True)

    def val_dataloader(self):
        if self.global_rank == 0:
            print('**********Start to Prepare the Validation Dataloader**********')
        if self.config.chinese:
            attri = 'gen'
        else:
            attri = 'gen_en'
        return create_dataloader(config=self.config, dataset=self.val_dataset,
                                 tokenizer=self.gen_tokenizer, attri=attri, shuffle=False)

    def predict_dataloader(self):
        if self.global_rank == 0:
            print('**********Start to Prepare the Predict Dataloader**********')
            print(f'**********The Top-P is {self.config.top_p}**********')
            print(f'**********The Repetition Penalty is {self.config.repetition_penalty}**********')
        return create_predict_dataloader(config=self.config, tokenizer=self.gen_tokenizer,
                                         rank=self.global_rank, attri='gen')

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
        if self.config.chinese:
            loss, _ = self.generator.forward(
                batch['total_num'].cuda(),
                batch['prompts_input_ids'].cuda(),
                batch['lengths_input_ids'].cuda(),
                batch['prompts_attention_mask'].cuda(),
            )
        else:
            loss, _ = self.generator.forward(
                batch['input_ids'].cuda(),
                batch['attention_mask'].cuda(),
                batch['lengths'].cuda(),
            )
            
        self.log('gen_train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_ids):
        torch.cuda.empty_cache()
        if self.config.chinese:
            loss, _ = self.generator.forward(
                batch['total_num'].cuda(),
                batch['prompts_input_ids'].cuda(),
                batch['lengths_input_ids'].cuda(),
                batch['prompts_attention_mask'].cuda()
            )
        else:
            loss, _ = self.generator.forward(
                batch['input_ids'].cuda(),
                batch['attention_mask'].cuda(),
                batch['lengths'].cuda(),
            )

        self.log('gen_val_loss', loss.item())
        self.log('gen_val_ppl', torch.exp(loss).item())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.cuda.empty_cache()

        self.generator.gen.to('cuda').eval()
        output_dict = sample_sequence_batch(
            model=self.generator.gen, context_tokens_tensor=batch['input_ids'].cuda(),
            context_length_tensor=batch['length_tensor'], repetition_penalty=self.config.repetition_penalty,
            max_out_seq=200, end_token_id=50000, temperature=1.0, top_k=self.config.top_k, top_p=self.config.top_p,
        )

        return output_dict
