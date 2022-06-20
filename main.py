import hydra
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         EarlyStopping)

from system.gen_system import GenSystem
from system.dis_system import DisSystem            
            

def set_trainer(config, steps, ckpt_callback, early_stopping, batch_size):
    lr_callback = LearningRateMonitor(logging_interval='step')
    if config.cycle < config.gen_anti_cyle:
        val_num = 5
    else:
        val_num = 20
    trainer = Trainer(
        default_root_dir=config.exp_dir,
        gpus=8,
        strategy=DeepSpeedStrategy(
            offload_optimizer=True,
            logging_batch_size_per_gpu=batch_size),
        precision=16,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        val_check_interval=val_num,
        callbacks=[lr_callback, ckpt_callback, early_stopping],
        max_steps=steps,
    )

    return trainer
    

def generator_cycle(config, gen_system):
    torch.cuda.empty_cache()
    gen_ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='gen_val_loss',
        mode='min',
        filename=f'generator_cycle_{config.cycle + 1}',
        dirpath=config.ckpt_model_path,
    )
    gen_early_stopping = EarlyStopping(
        monitor='gen_val_loss',
        patience=3,
        mode='min'
    )
    gen_trainer = set_trainer(
        config=config,
        steps=int(config.gen_train_steps), 
        ckpt_callback=gen_ckpt_callback, 
        early_stopping=gen_early_stopping,
        batch_size=config.gen_batch_size
    )

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
        patience=3,
        mode='max'
    )
    dis_trainer = set_trainer(
        config=config,
        steps=int(config.dis_train_steps),
        ckpt_callback=dis_ckpt_callback,
        early_stopping=dis_early_stopping,
        batch_size=config.dis_batch_size
    )
    
    dis_trainer.fit(dis_system)
    dis_system.judge_similarity()


@hydra.main(config_path='./', config_name='hyper_parameter')
def run(config):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed_everything(config.seed)
    
    gen_system = GenSystem(config)
    dis_system = DisSystem(config)
    
    for idx in range(0, config.cycle_nums):
        config.cycle = idx
        print('Cycle: {}'.format(config.cycle))

        generator_cycle(config, gen_system)
        discriminator_cycle(config, dis_system)


if __name__ == '__main__':
    run()
