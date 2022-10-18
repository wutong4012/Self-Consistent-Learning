import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
from omegaconf import OmegaConf
from collections import defaultdict

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping)

from utils import check_acc
from system.gen_system import GenSystem
from system.dis_system import DisSystem     
from data_utlis.dist_gather import all_gather 
from data_utlis.predict_dataset import gen_postprocess, dis_postprocess      
            

def set_trainer(config, ckpt_callback, early_stopping):
    trainer = Trainer(
        default_root_dir=config.exp_dir,
        gpus=1,
        strategy=DeepSpeedStrategy(
            offload_optimizer=True,
            logging_batch_size_per_gpu=1),
        precision=16,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1, 
        callbacks=[ckpt_callback, early_stopping],
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

    gen_output = concat_data(all_gather(gen_trainer.predict(gen_system)))
    gen_postprocess(gen_output, gen_system.gen_tokenizer, config)


def discriminator_cycle(config):
    dis_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='dis_val_loss',
        mode='min',
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

    dis_output = concat_data(all_gather(dis_trainer.predict(dis_system)))
    dis_postprocess(dis_output, config)


def server_func(data_path, dis_ckpt_path):
    """_summary_

    Args:
        data_path (_type_): 文件夹下包含train.json和test_public.json
        dis_ckpt_path (_type_): 保存的finetune.bin文件
    """
    config = OmegaConf.load('./hyper_parameter.yaml')
    config.data_path = data_path
    config.dis_ckpt_path = dis_ckpt_path
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed_everything(config.seed)
    
    # config.ckpt_model_path += str(config.idx)
    # config.sim_data_path += str(config.idx)
    
    curr_acc = 0
    for idx in range(config.cycle, config.cycle_num):
        config.cycle = idx
        print('**********Cycle: {}**********'.format(config.cycle))

        generator_cycle(config)
        gc.collect()
        discriminator_cycle(config)
        gc.collect()
        
        if config.cycle > 0:
            is_continue, curr_acc = check_acc(config, last_acc=curr_acc)
            if not is_continue:
                break
    
    print('Cylce Over!')
    return config.ckpt_model_path + \
        f'/discriminator_cycle_{config.cycle-1}.ckpt/checkpoint/mp_rank_00_model_states.pt'


print(server_func(data_path='/cognitive_comp/wutong/source/sim_data/raw_data/bustm',
                  dis_ckpt_path = '/cognitive_comp/wutong/finetune_large_dev.bin'))