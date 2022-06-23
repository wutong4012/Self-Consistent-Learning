from re import L
import hydra
import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         EarlyStopping)
from yaml import parse

from system.gen_system import GenSystem
from system.dis_system import DisSystem            
            

def set_trainer(config, ckpt_callback, early_stopping):
    lr_callback = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        default_root_dir=config.exp_dir,
        gpus=8,
        strategy=DeepSpeedStrategy(
            offload_optimizer=True,
            logging_batch_size_per_gpu=1),
        precision=16,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[lr_callback, ckpt_callback, early_stopping],
        max_epochs=config.epochs,
    )

    return trainer
    

def generator_cycle(config, gen_system):
    gen_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='gen_val_loss',
        mode='min',
        filename=f'generator_cycle_{config.cycle + 1}',
        dirpath=config.ckpt_model_path,
    )
    gen_early_stopping = EarlyStopping(
        monitor='gen_val_loss',
        patience=2,
        mode='min'
    )
    gen_trainer = set_trainer(
        config=config, 
        ckpt_callback=gen_ckpt_callback, 
        early_stopping=gen_early_stopping,
    )

    torch.cuda.empty_cache()
    gen_trainer.fit(gen_system)
    gen_system.generate_samples()


def discriminator_cycle(config, dis_system):
    dis_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='dis_f1_score',
        mode='max',
        filename=f'discriminator_cycle_{config.cycle + 1}',
        dirpath=config.ckpt_model_path,
    )
    dis_early_stopping = EarlyStopping(
        monitor='dis_f1_score',
        patience=2,
        mode='max'
    )
    dis_trainer = set_trainer(
        config=config,
        ckpt_callback=dis_ckpt_callback,
        early_stopping=dis_early_stopping,
    )
    
    torch.cuda.empty_cache()
    dis_trainer.fit(dis_system)
    dis_system.judge_similarity()


@hydra.main(config_path='./', config_name='hyper_parameter')
def run(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed_everything(config.seed)
    
    gen_system = GenSystem(config)
    dis_system = DisSystem(config)

    print('Cycle: {}'.format(config.cycle))
    generator_cycle(config, gen_system)
    discriminator_cycle(config, dis_system)


if __name__ == '__main__':
    run()
