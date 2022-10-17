import os
import gc
import hydra
from collections import defaultdict

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         EarlyStopping)

from system.gen_system import GenSystem
from system.dis_system import DisSystem     
from data_utlis.dist_gather import all_gather 
from data_utlis.predict_dataset import gen_postprocess, dis_postprocess      
            

def set_trainer(config, ckpt_callback, early_stopping):
    lr_callback = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        default_root_dir=config.exp_dir,
        gpus=1,  ##
        strategy=DeepSpeedStrategy(
            offload_optimizer=True,
            logging_batch_size_per_gpu=1),
        precision=16,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,  ##
        # val_check_interval=500,
        callbacks=[lr_callback, ckpt_callback, early_stopping],
    )

    return trainer


def concat_data(raw_list):  # List[Dict]<-(world_size, batch_num)
    concate_output = defaultdict(list)
    for item_list in raw_list:
        for batch in item_list:
            for key in batch.keys():
                concate_output[key].extend(batch[key])
    
    return concate_output


def generator_cycle(config):
    gen_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='gen_val_loss',
        mode='min',
        filename=f'generator_cycle_{config.cycle + 1}',
        dirpath=config.ckpt_model_path,
    )
    gen_early_stopping = EarlyStopping(
        monitor='gen_val_loss',
        patience=config.es_patience,
        mode='min'
    )
    gen_trainer = set_trainer(
        config=config, 
        ckpt_callback=gen_ckpt_callback, 
        early_stopping=gen_early_stopping,
    )
    gen_system = GenSystem(config)

    torch.cuda.empty_cache()
    if config.cycle != -1:
        gen_trainer.fit(gen_system)

    if not config.pretrain_gen:
        gen_output = concat_data(all_gather(gen_trainer.predict(gen_system)))
        gen_postprocess(gen_output, gen_system.gen_tokenizer, 
                        config, gen_system.global_rank)


def discriminator_cycle(config):
    dis_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='dis_val_loss',  # dis_f1_score / dis_val_loss
        mode='min',  # max / min
        filename=f'discriminator_cycle_{config.cycle + 1}',
        dirpath=config.ckpt_model_path,
    )
    dis_early_stopping = EarlyStopping(
        monitor='dis_val_loss',
        patience=config.es_patience,
        mode='min'
    )
    dis_trainer = set_trainer(
        config=config,
        ckpt_callback=dis_ckpt_callback,
        early_stopping=dis_early_stopping,
    )
    dis_system = DisSystem(config)
    
    torch.cuda.empty_cache()
    if config.cycle != -1:
        dis_trainer.fit(dis_system)

    if not config.pretrain_dis:
        dis_output = concat_data(all_gather(dis_trainer.predict(dis_system)))
        dis_postprocess(dis_output, config, dis_system.global_rank)


@hydra.main(config_path='./', config_name='hyper_parameter')
def run(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed_everything(config.seed)
    
    config.ckpt_model_path += str(config.idx)
    config.sim_data_path += str(config.idx)
    config.cache_data_path += str(config.idx)
    if not os.path.exists(config.cache_data_path):
        os.mkdir(config.cache_data_path)
    
    for idx in range(config.cycle, config.cycle_num):
        config.cycle = idx
        print('**********Cycle: {}**********'.format(config.cycle))

        if not config.pretrain_dis:
            generator_cycle(config)
            gc.collect()
        if not config.pretrain_gen:
            discriminator_cycle(config)
            gc.collect()


if __name__ == '__main__':
    run()
