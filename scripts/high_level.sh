mkdir -p ./logs/high_level
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'mps' \
    >./logs/high_level/ETHUSDT.log 2>&1 &