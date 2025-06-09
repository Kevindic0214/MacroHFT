 #!/bin/bash

echo "🌈 啟動 Rainbow DQN Low Level 訓練..."

# 創建日誌目錄
mkdir -p ./logs/low_level/ETHUSDT/

# Rainbow DQN 優化參數
RAINBOW_PARAMS="--n_step 3 --lr 6.25e-5 --batch_size 32 --epsilon_start 0.1 --epsilon_end 0.01 --tau 0.005"

nohup python -u RL/agent/low_level.py --alpha 1 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_1 $RAINBOW_PARAMS \
    >./logs/low_level/ETHUSDT/slope_1_rainbow.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 4 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_2 $RAINBOW_PARAMS \
    >./logs/low_level/ETHUSDT/slope_2_rainbow.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 0 --clf 'slope' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_3 $RAINBOW_PARAMS \
    >./logs/low_level/ETHUSDT/slope_3_rainbow.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 4 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_1 $RAINBOW_PARAMS \
    >./logs/low_level/ETHUSDT/vol_1_rainbow.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_2 $RAINBOW_PARAMS \
    >./logs/low_level/ETHUSDT/vol_2_rainbow.log 2>&1 &

nohup python -u RL/agent/low_level.py --alpha 1 --clf 'vol' --dataset 'ETHUSDT' --device 'cuda:0' \
    --label label_3 $RAINBOW_PARAMS \
    >./logs/low_level/ETHUSDT/vol_3_rainbow.log 2>&1 &
