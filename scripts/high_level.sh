#!/bin/bash

mkdir -p ./logs/high_level/

# 傳統軟混合策略訓練
echo "開始傳統軟混合策略訓練..."
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:9' \
    --use_dynamic_mixing False \
    --exp 'traditional_soft_mixing' \
    >./logs/high_level/ETHUSDT_traditional.log 2>&1 &

# 動態混合策略訓練（默認參數）
echo "開始動態混合策略訓練（默認參數）..."
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:7' \
    --use_dynamic_mixing True \
    --mixing_loss_weight 0.1 \
    --strategy_consistency_weight 0.05 \
    --exp 'dynamic_mixing_default' \
    >./logs/high_level/ETHUSDT_dynamic_default.log 2>&1 &

# 動態混合策略訓練（激進切換）
echo "開始動態混合策略訓練（激進切換）..."
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:8' \
    --use_dynamic_mixing True \
    --mixing_loss_weight 0.2 \
    --strategy_consistency_weight 0.01 \
    --exp 'dynamic_mixing_aggressive' \
    >./logs/high_level/ETHUSDT_dynamic_aggressive.log 2>&1 &

# 動態混合策略訓練（保守切換）
echo "開始動態混合策略訓練（保守切換）..."
nohup python -u RL/agent/high_level.py --dataset 'ETHUSDT' --device 'cuda:9' \
    --use_dynamic_mixing True \
    --mixing_loss_weight 0.05 \
    --strategy_consistency_weight 0.1 \
    --exp 'dynamic_mixing_conservative' \
    >./logs/high_level/ETHUSDT_dynamic_conservative.log 2>&1 &

echo "所有訓練任務已啟動，請查看日誌文件監控進度"
echo "日誌文件位置："
echo "  - 傳統軟混合: ./logs/high_level/ETHUSDT_traditional.log"
echo "  - 動態混合默認: ./logs/high_level/ETHUSDT_dynamic_default.log"  
echo "  - 動態混合激進: ./logs/high_level/ETHUSDT_dynamic_aggressive.log"
echo "  - 動態混合保守: ./logs/high_level/ETHUSDT_dynamic_conservative.log"