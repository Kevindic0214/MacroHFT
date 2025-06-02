#!/bin/bash

mkdir -p ./logs/high_level/

nohup python -u RL/agent/high_level.py \
    --dataset "ETHUSDT" \
    --device "cuda:0" \
    --lr 1e-4 \
    --batch_size 512 \
    --epsilon_start 0.7 \
    --epsilon_end 0.1 \
    --decay_length 6 \
    --alpha 0.1 \
    --beta 0.5 \
    --hyperagent_hidden_dim 64 \
    --num_quantiles 51 \
    --epoch_number 8 \
    --exp "exp_suggested_params" > ./logs/high_level/ETHUSDT.log 2>&1 &