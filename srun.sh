#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 


## zh zero-shot
# for idx in {0..2}
# do

#     if [ "$idx" == "0" ]; then
#         data_name='qqp'
#         discriminator_zh='roberta_large'
#         max_thre0=0.9
#         min_thre0=0.8
#         max_thre1=0.95
#         min_thre1=0.9
#         max_dis_thre=0.8
#         min_dis_thre=0.8
#         sentence_num=3000
#         zero_shot=1
#         chinese=1
#         cycle_num=11
#         dis_batch_size=64
#         pre_gen_bs=512
#         pre_dis_bs=384
#         consistency=1
#         add_thre=0

#     elif [ "$idx" == "1" ]; then
#         data_name='afqmc'
#         discriminator_zh='roberta_large'
#         max_thre0=0.95
#         min_thre0=0.8
#         max_thre1=0.95
#         min_thre1=0.8
#         max_dis_thre=0.95
#         min_dis_thre=0.8
#         sentence_num=6000
#         zero_shot=1
#         chinese=1
#         cycle_num=9
#         dis_batch_size=64
#         pre_gen_bs=512
#         pre_dis_bs=384
#         consistency=1
#         add_thre=0
    
#     elif [ "$idx" == "2" ]; then
#         data_name='chip'
#         discriminator_zh='roberta_large'
#         max_thre0=0.9
#         min_thre0=0.8
#         max_thre1=0.9
#         min_thre1=0.8
#         max_dis_thre=0.9
#         min_dis_thre=0.8
#         sentence_num=6000
#         zero_shot=1
#         chinese=1
#         cycle_num=11
#         dis_batch_size=64
#         pre_gen_bs=512
#         pre_dis_bs=384
#         consistency=1
#         add_thre=0

#     fi

#     echo "RUN test $idx"
#     srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++data_name=$data_name ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 \
#         ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre ++sentence_num=$sentence_num ++min_thre0=$min_thre0 ++min_thre1=$min_thre1 ++zero_shot=$zero_shot \
#         ++discriminator_zh=$discriminator_zh ++cycle_num=$cycle_num ++chinese=$chinese ++dis_batch_size=$dis_batch_size ++consistency=$consistency ++add_thre=$add_thre \
#         ++pre_gen_bs=$pre_gen_bs ++pre_dis_bs=$pre_dis_bs
#     echo "END test $idx"

# done

### zh fine-tune
# for idx in {11..12}
# do
#     # if [ "$idx" == "10" ]; then
#     #     data_name='afqmc'
#     #     discriminator_zh='roformer_large'
#     #     max_thre0=0.98
#     #     min_thre0=0.9
#     #     max_thre1=0.98
#     #     min_thre1=0.9
#     #     max_dis_thre=0.98
#     #     min_dis_thre=0.9
#     #     sentence_num=6000
#     #     zero_shot=0
#     #     chinese=1
#     #     cycle_num=10
#     #     dis_batch_size=32
#     #     pre_gen_bs=512
#     #     pre_dis_bs=256
#     #     learning_rate=5e-6
    
#     if [ "$idx" == "11" ]; then
#         data_name='chip'
#         discriminator_zh='roberta_large'
#         max_thre0=0.98
#         min_thre0=0.7
#         max_thre1=0.98
#         min_thre1=0.7
#         max_dis_thre=0.98
#         min_dis_thre=0.7
#         sentence_num=6000
#         zero_shot=0
#         chinese=1
#         cycle_num=10
#         dis_batch_size=64
#         pre_gen_bs=512
#         pre_dis_bs=384
#         learning_rate=5e-6
    
#     elif [ "$idx" == "12" ]; then
#         data_name='qqp'
#         discriminator_zh='roformer_large'
#         max_thre0=0.84
#         min_thre0=0.6
#         max_thre1=0.98
#         min_thre1=0.9
#         max_dis_thre=0.98
#         min_dis_thre=0.9
#         sentence_num=3000
#         zero_shot=0
#         chinese=1
#         cycle_num=10
#         dis_batch_size=32
#         pre_gen_bs=512
#         pre_dis_bs=256
#         learning_rate=5e-6

#     fi

#     echo "RUN test $idx"
#     srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++data_name=$data_name ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 \
#         ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre ++sentence_num=$sentence_num ++min_thre0=$min_thre0 ++min_thre1=$min_thre1 ++zero_shot=$zero_shot \
#         ++discriminator_zh=$discriminator_zh ++cycle_num=$cycle_num ++chinese=$chinese ++dis_batch_size=$dis_batch_size ++learning_rate=$learning_rate \
#         ++pre_gen_bs=$pre_gen_bs ++pre_dis_bs=$pre_dis_bs
#     echo "END test $idx"

# done


## en train_gen
# for idx in {10..10}
# do
#     if [ "$idx" == "10" ]; then
#         es_patience=3
#         pretrain_gen=1
#         warm_up_model=0
#         cycle=0
#         cycle_num=1
#         gen_train_steps=50000
#         warmup_steps=5000

#     fi

#     echo "RUN test $idx"
#     srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py ++data_name=$data_name ++idx=$idx \
#     ++es_patience=$es_patience ++pretrain_gen=$pretrain_gen ++warm_up_model=$warm_up_model \
#     ++cycle=$cycle ++cycle_num=$cycle_num ++gen_train_steps=$gen_train_steps ++warmup_steps=$warmup_steps
#     echo "END test $idx"

# done


### en 
# for idx in {10..10}
# do
#     if [ "$idx" == "10" ]; then
#         data_name='mrpc'
#         discriminator_en='albert_xxlarge'
#         max_thre0=0.95
#         min_thre0=0.8
#         max_thre1=0.95
#         min_thre1=0.8
#         max_dis_thre=0.95
#         min_dis_thre=0.8
#         sentence_num=4000
#         top_p=0.95
#         cycle_num=10
#         learning_rate=2e-5
#         add_thre=0.05
#         zero_shot=1
#         chinese=0
#         consistency=0
#         dis_use_label=0
#         opt_name='opt-2.7b'

#     fi

#     echo "RUN test $idx"
#     srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++data_name=$data_name ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 \
#         ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre ++sentence_num=$sentence_num ++min_thre0=$min_thre0 ++min_thre1=$min_thre1 ++zero_shot=$zero_shot \
#         ++discriminator_en=$discriminator_en ++cycle_num=$cycle_num ++chinese=$chinese ++top_p=$top_p ++learning_rate=$learning_rate ++consistency=$consistency \
#         ++add_thre=$add_thre ++opt_name=$opt_name ++dis_use_label=$dis_use_label
#     echo "END test $idx"

# done


for idx in {1..2}
do
    if [ "$idx" == "0" ]; then
        data_name='mrpc'
        discriminator_en='albert_xxlarge'
        max_thre0=0.95
        min_thre0=0.8
        max_thre1=0.95
        min_thre1=0.8
        max_dis_thre=0.95
        min_dis_thre=0.8
        sentence_num=4000
        top_p=0.95
        cycle_num=10
        learning_rate=2e-5
        add_thre=0.05
        zero_shot=1
        chinese=0
        opt_name='opt-2.7b'

    elif [ "$idx" == "1" ]; then
        data_name='mrpc'
        discriminator_en='albert_xxlarge'
        max_thre0=0.95
        min_thre0=0.8
        max_thre1=0.95
        min_thre1=0.8
        max_dis_thre=0.95
        min_dis_thre=0.8
        sentence_num=4000
        top_p=0.95
        cycle_num=10
        learning_rate=2e-5
        add_thre=0.05
        zero_shot=1
        chinese=0
        opt_name='opt-350m'

    elif [ "$idx" == "2" ]; then
        data_name='mrpc'
        discriminator_en='albert_xxlarge'
        max_thre0=0.8
        min_thre0=0.6
        max_thre1=0.8
        min_thre1=0.6
        max_dis_thre=0.8
        min_dis_thre=0.6
        sentence_num=3000
        top_p=0.95
        cycle_num=10
        learning_rate=5e-6
        add_thre=1.0
        zero_shot=0
        chinese=0
        opt_name='opt-350m'

    fi

    echo "RUN test $idx"
    srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++data_name=$data_name ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 \
        ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre ++sentence_num=$sentence_num ++min_thre0=$min_thre0 ++min_thre1=$min_thre1 ++zero_shot=$zero_shot \
        ++discriminator_en=$discriminator_en ++cycle_num=$cycle_num ++chinese=$chinese ++top_p=$top_p ++learning_rate=$learning_rate \
        ++add_thre=$add_thre ++opt_name=$opt_name
    echo "END test $idx"

done


## train_dis
# for idx in {23..23}
# do
#     if [ "$idx" == "23" ]; then
#         data_name='qqp'
#         discriminator_zh='albert_xxlarge'
#         es_patience=3
#         pretrain_dis=1
#         warm_up_model=0
#         cycle=0
#         cycle_num=1
#         dis_train_steps=1000
#         warmup_steps=30
#         dis_batch_size=128
#         learning_rate=5e-6
#         zero_shot=0
#         chinese=1

#     fi

#     echo "RUN test $idx"
#     # srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err 
#     python main.py ++data_name=$data_name ++idx=$idx \
#     ++discriminator_zh=$discriminator_zh ++es_patience=$es_patience ++pretrain_dis=$pretrain_dis ++warm_up_model=$warm_up_model \
#     ++cycle=$cycle ++cycle_num=$cycle_num ++dis_train_steps=$dis_train_steps ++warmup_steps=$warmup_steps ++chinese=$chinese \
#     dis_batch_size=$dis_batch_size ++zero_shot=$zero_shot ++learning_rate=$learning_rate
#     echo "END test $idx"

# done
