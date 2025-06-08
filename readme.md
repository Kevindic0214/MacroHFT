# MacroHFT
This is the official implementation of the KDD 2024 "MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading".
https://arxiv.org/abs/2406.14537

To run the demo code:

You may first download the dataset from Google Drive:

https://drive.google.com/drive/folders/1AYHy-wUV0IwPoA7E1zvMRPL3wK0tPNiY?usp=drive_link

and put the folder under data folder.

## Step 1
Run scripts/decomposition.sh for data decomposition and labeling. 
## Step 2
Run scripts/low_level.sh for low-level policy optimization. 

Update: We now provide trained model checkpoints for sub-agents, which can be directly used to train meta-policy.
## Step 3
Run scripts/high_level.sh for meta-policy optimization.

## Project Structure

### Performance Folder
The `performance` folder contains tools and results for evaluating the MacroHFT model:
- `performance_analyzer.py`: Main script for analyzing trading performance metrics
- `combined_performance_analysis`: Contains combined performance results across different models
- `performance_analysis_output`: Detailed performance metrics, charts, and statistics
- Various model result folders (`macrohft_result`, `multipatchformer_result`, `qrdqn_result`, `rainbowdqn_result`): Contains performance data for different model implementations

### Baseline Models

#### ATR Baseline
The `atr_baseline` folder contains an implementation of the Average True Range (ATR) trading strategy as a baseline:
- `atr_trend_strategy_backtest.py`: Implementation of the ATR trend-following strategy
- `strategy_performance.py`: Script for evaluating the ATR strategy performance
- `test_performance_report`: Contains performance reports and metrics for the ATR strategy

#### PPO Baseline
The `ppo_baseline` folder contains an implementation of the Proximal Policy Optimization (PPO) reinforcement learning algorithm as a baseline:
- `stock_trading.py`: Implementation of the PPO-based trading environment and agent
- `strategy_performance.py`: Script for evaluating the PPO strategy performance
- `performance_report`: Contains performance reports and metrics for the PPO strategy
- `ppo_qqq_model.zip`: Pre-trained PPO model checkpoint
