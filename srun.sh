#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 


for idx in {0..4}
do
    if [ "$idx" == "0" ]; then
        max_thre0=0.95
        min_thre0=0.9
        max_thre1=0.95
        min_thre1=0.9
        max_dis_thre=0.95
        min_dis_thre=0.85
        sentence_num=3000
        add_thre=0.01

    elif [ "$idx" == "1" ]; then
        max_thre0=0.95
        min_thre0=0.9
        max_thre1=0.95
        min_thre1=0.9
        max_dis_thre=0.95
        min_dis_thre=0.85
        sentence_num=4000
        add_thre=0.01

    elif [ "$idx" == "2" ]; then
        max_thre0=0.95
        min_thre0=0.9
        max_thre1=0.95
        min_thre1=0.9
        max_dis_thre=0.95
        min_dis_thre=0.85
        sentence_num=2000
        add_thre=0.01

    elif [ "$idx" == "3" ]; then
        max_thre0=0.95
        min_thre0=0.9
        max_thre1=0.95
        min_thre1=0.9
        max_dis_thre=0.95
        min_dis_thre=0.85
        sentence_num=2500
        add_thre=0.01

    elif [ "$idx" == "4" ]; then
        max_thre0=0.95
        min_thre0=0.9
        max_thre1=0.95
        min_thre1=0.9
        max_dis_thre=0.95
        min_dis_thre=0.85
        sentence_num=3500
        add_thre=0.01

    fi

    echo "RUN test $idx"
    srun --gres=gpu:1 -o ./job_out/%x-%j-$idx.log -e ./job_out/%x-%j-$idx.err python sim_gen_server.py ++idx=$idx ++max_thre0=$max_thre0 ++max_thre1=$max_thre1 \
        ++max_dis_thre=$max_dis_thre ++min_dis_thre=$min_dis_thre ++sentence_num=$sentence_num ++min_thre0=$min_thre0 ++min_thre1=$min_thre1 ++add_thre=$add_thre
    echo "END test $idx"

done

