#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=20
# -e job_out/%x-%j.err -o job_out/%x-%j.log
srun python sim_dis_load.py
