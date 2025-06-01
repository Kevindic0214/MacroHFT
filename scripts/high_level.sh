#!/bin/bash

mkdir -p ./logs/high_level/

nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:0' --num_quantiles 51 \
    >./logs/high_level/ETHUSDT.log 2>&1 &