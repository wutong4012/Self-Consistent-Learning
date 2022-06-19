#!/bin/bash

# -N 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=30 -x 

srun -e job_out/%x-%j.err -o job_out/%x-%j.log python main.py
