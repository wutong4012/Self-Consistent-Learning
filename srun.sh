#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 

for idx in {10..12}
do
    if [ "$idx" == "10" ]; then
        std_scale=1.2
        max_thre0=0.8
        max_thre1=0.8
        max_dis_thre=0.9
        min_dis_thre=0.9

    elif [ "$idx" == "11" ]; then
        std_scale=1.0
        max_thre0=0.8
        max_thre1=0.8
        max_dis_thre=0.9
        min_dis_thre=0.9

    elif [ "$idx" == "12" ]; then
        std_scale=0.8
        max_thre0=0.8
        max_thre1=0.8
        max_dis_thre=0.9
        min_dis_thre=0.9

    fi

    echo "RUN test $idx"
    srun --gres=gpu:8 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python main.py  ++idx=$idx ++std_scale=$std_scale ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre
    echo "END test $idx"

done

# srun --gres=gpu:8 -o ./job_out/%x-%j.log -e ./job_out/%x-%j.err python main.py
