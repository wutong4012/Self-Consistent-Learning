#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 


# for idx in {10..12}
# do
#     if [ "$idx" == "10" ]; then
#         data_name='qqp'
#         max_thre0=0.9
#         min_thre0=0.7
#         max_thre1=0.9
#         min_thre1=0.7
#         max_dis_thre=0.9
#         min_dis_thre=0.7
#         sentence_num=6000
#         zero_shot=1
#         cycle_num=20
#         repetition_penalty=1.0

#     elif [ "$idx" == "11" ]; then
#         data_name='qqp'
#         max_thre0=0.9
#         min_thre0=0.7
#         max_thre1=0.9
#         min_thre1=0.7
#         max_dis_thre=0.9
#         min_dis_thre=0.7
#         sentence_num=3000
#         zero_shot=0
#         cycle_num=10
#         repetition_penalty=1.3

#     elif [ "$idx" == "12" ]; then
#         data_name='qqp'
#         max_thre0=0.7
#         min_thre0=0.7
#         max_thre1=0.7
#         min_thre1=0.7
#         max_dis_thre=0.7
#         min_dis_thre=0.7
#         sentence_num=3000
#         zero_shot=0
#         cycle_num=10
#         repetition_penalty=1.2


for idx in {0..0}
do
    if [ "$idx" == "0" ]; then
        data_name='afqmc'
        max_thre0=0.9
        min_thre0=0.7
        max_thre1=0.9
        min_thre1=0.7
        max_dis_thre=0.7
        min_dis_thre=0.7
        sentence_num=6000
        zero_shot=1
        repetition_penalty=1.0
        cycle_num=10

    # elif [ "$idx" == "1" ]; then
    #     data_name='qqp'
    #     max_thre0=0.8
    #     min_thre0=0.6
    #     max_thre1=0.8
    #     min_thre1=0.6
    #     max_dis_thre=0.9
    #     min_dis_thre=0.9
    #     sentence_num=3000
    #     zero_shot=0
    #     repetition_penalty=1.2
    #     cycle_num=10

    # elif [ "$idx" == "2" ]; then
    #     data_name='qqp'
    #     max_thre0=0.6
    #     min_thre0=0.5
    #     max_thre1=0.9
    #     min_thre1=0.8
    #     max_dis_thre=0.9
    #     min_dis_thre=0.9
    #     sentence_num=3000
    #     zero_shot=0
    #     repetition_penalty=1.0
    #     cycle_num=10

    fi

    echo "RUN test $idx"
    srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++data_name=$data_name ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 \
        ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre ++sentence_num=$sentence_num ++min_thre0=$min_thre0 ++min_thre1=$min_thre1 ++zero_shot=$zero_shot \
        ++repetition_penalty=$repetition_penalty ++cycle_num=$cycle_num
    echo "END test $idx"

done

# srun --gres=gpu:8 -o ./job_out/%x-%j.log -e ./job_out/%x-%j.err python main.py ++data_name='qqp'
