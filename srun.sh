#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 

for cycle in {0..10}
do
        echo "RUN cycle $cycle"
        srun -o ./job_out/%x-%j-$cycle.log -e ./job_out/%x-%j-$cycle.err python main.py +cycle=$cycle
        echo "END cycle $cycle"
done
