#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=20

srun -e job_out/%x-%j.err -o job_out/%x-%j.log python sim_gen_load.py
