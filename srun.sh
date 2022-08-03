#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 

for idx in {0..0}
do
    if [ "$idx" == "0" ]; then
        std_scale=1.2
        max_thre0=0.8
        max_thre1=0.8
        max_dis_thre=0.8
    
    elif [ "$idx" == "10" ]; then
        std_scale=1.2
        max_thre0=0.7
        max_thre1=0.7
        max_dis_thre=0.7

    # elif [ "$idx" == "2" ]; then
    #     std_scale=2.0
    #     max_thre0=0.8
    #     max_thre1=0.8
    #     max_dis_thre=0.8
    
    # elif [ "$idx" == "3" ]; then
    #     std_scale=1.5
    #     max_thre0=0.6
    #     max_thre1=0.6
    #     max_dis_thre=0.6
    fi

    echo "RUN test $idx"
    srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++std_scale=$std_scale ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 ++max_dis_thre=$max_dis_thre
    echo "END test $idx"

done

# srun --gres=gpu:8 -o ./job_out/%x-%j.log -e ./job_out/%x-%j.err python main.py
