#!/bin/bash

mkdir -p ./logs/high_level/
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:6' \
    >>./logs/high_level/ETHUSDT_$TIMESTAMP.log 2>&1 &