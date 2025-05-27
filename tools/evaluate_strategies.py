import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
import torch
from collections import defaultdict

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StrategyEvaluator:
    def __init__(self, result_base_path="./result/high_level/ETHUSDT"):
        self.result_base_path = result_base_path
        self.strategies = {
            'traditional_soft_mixing': '傳統軟混合',
            'dynamic_mixing_default': '動態混合（默認）',
            'dynamic_mixing_aggressive': '動態混合（激進）',
            'dynamic_mixing_conservative': '動態混合（保守）'
        }
        
    def load_results(self, strategy_name):
        """載入特定策略的結果"""
        strategy_path = os.path.join(self.result_base_path, strategy_name)
        
        try:
            # 載入測試結果
            action = np.load(os.path.join(strategy_path, "action.npy"))
            reward = np.load(os.path.join(strategy_path, "reward.npy"))
            final_balance = np.load(os.path.join(strategy_path, "final_balance.npy"))
            required_money = np.load(os.path.join(strategy_path, "require_money.npy"))
            commission_fee = np.load(os.path.join(strategy_path, "commission_fee_history.npy"))
            
            return {
                'action': action,
                'reward': reward,
                'final_balance': final_balance,
                'required_money': required_money,
                'commission_fee': commission_fee,
                'return_rate': final_balance / required_money if required_money != 0 else 0
            }
        except FileNotFoundError as e:
            print(f"警告：無法載入策略 {strategy_name} 的結果: {e}")
            return None
    
    def calculate_metrics(self, results):
        """計算詳細的性能指標"""
        if results is None:
            return None
            
        rewards = results['reward'][0] if len(results['reward'].shape) > 1 else results['reward']
        actions = results['action'][0] if len(results['action'].shape) > 1 else results['action']
        
        # 基礎指標
        total_return = results['return_rate']
        final_balance = results['final_balance']
        required_money = results['required_money']
        commission_fee = results['commission_fee']
        
        # 風險指標
        returns = np.array(rewards)
        volatility = np.std(returns)
        sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown)
        
        # 交易頻率
        action_changes = np.sum(np.diff(actions) != 0)
        trading_frequency = action_changes / len(actions)
        
        # 勝率
        positive_returns = np.sum(returns > 0)
        win_rate = positive_returns / len(returns)
        
        # 盈虧比
        positive_mean = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        negative_mean = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
        profit_loss_ratio = abs(positive_mean / negative_mean) if negative_mean != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'final_balance': final_balance,
            'required_money': required_money,
            'commission_fee': commission_fee[0] if hasattr(commission_fee, '__len__') else commission_fee,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trading_frequency': trading_frequency,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': action_changes
        }
    
    def load_tensorboard_logs(self, strategy_name):
        """載入TensorBoard日誌數據"""
        log_path = os.path.join(self.result_base_path, strategy_name, "seed_12345", "log")
        
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            ea = EventAccumulator(log_path)
            ea.Reload()
            
            # 獲取標量數據
            scalar_tags = ea.Tags()['scalars']
            
            logs = {}
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                logs[tag] = [(s.step, s.value) for s in scalar_events]
            
            return logs
        except Exception as e:
            print(f"警告：無法載入 {strategy_name} 的TensorBoard日誌: {e}")
            return {}
    
    def compare_strategies(self):
        """比較所有策略的性能"""
        all_results = {}
        all_metrics = {}
        
        for strategy_key, strategy_name in self.strategies.items():
            print(f"載入策略: {strategy_name}")
            results = self.load_results(strategy_key)
            if results is not None:
                all_results[strategy_name] = results
                all_metrics[strategy_name] = self.calculate_metrics(results)
        
        return all_results, all_metrics
    
    def create_comparison_table(self, all_metrics):
        """創建性能比較表"""
        if not all_metrics:
            print("沒有可用的策略結果進行比較")
            return None
            
        df_data = []
        for strategy_name, metrics in all_metrics.items():
            if metrics is not None:
                df_data.append({
                    '策略': strategy_name,
                    '總收益率': f"{metrics['total_return']:.4f}",
                    '最終餘額': f"{metrics['final_balance']:.2f}",
                    '所需資金': f"{metrics['required_money']:.2f}",
                    '手續費': f"{metrics['commission_fee']:.2f}",
                    '波動率': f"{metrics['volatility']:.4f}",
                    '夏普比率': f"{metrics['sharpe_ratio']:.4f}",
                    '最大回撤': f"{metrics['max_drawdown']:.4f}",
                    '交易頻率': f"{metrics['trading_frequency']:.4f}",
                    '勝率': f"{metrics['win_rate']:.4f}",
                    '盈虧比': f"{metrics['profit_loss_ratio']:.2f}",
                    '總交易次數': metrics['total_trades']
                })
        
        df = pd.DataFrame(df_data)
        return df
    
    def plot_performance_comparison(self, all_results, save_path="./analysis"):
        """繪製性能比較圖"""
        if not all_results:
            print("沒有可用數據進行繪圖")
            return
            
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 累積收益對比
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        for strategy_name, results in all_results.items():
            rewards = results['reward'][0] if len(results['reward'].shape) > 1 else results['reward']
            cumulative_returns = np.cumsum(rewards)
            plt.plot(cumulative_returns, label=strategy_name, linewidth=2)
        plt.title('累積收益對比', fontsize=14, fontweight='bold')
        plt.xlabel('時間步')
        plt.ylabel('累積收益')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 收益率分佈
        plt.subplot(2, 3, 2)
        return_rates = [results['return_rate'] for results in all_results.values()]
        strategy_names = list(all_results.keys())
        bars = plt.bar(range(len(strategy_names)), return_rates, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('總收益率對比', fontsize=14, fontweight='bold')
        plt.xlabel('策略')
        plt.ylabel('收益率')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
        
        # 在柱狀圖上添加數值標籤
        for i, (bar, rate) in enumerate(zip(bars, return_rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{rate:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. 夏普比率對比
        plt.subplot(2, 3, 3)
        sharpe_ratios = []
        for results in all_results.values():
            rewards = results['reward'][0] if len(results['reward'].shape) > 1 else results['reward']
            volatility = np.std(rewards)
            sharpe = np.mean(rewards) / volatility if volatility > 0 else 0
            sharpe_ratios.append(sharpe)
        
        bars = plt.bar(range(len(strategy_names)), sharpe_ratios, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('夏普比率對比', fontsize=14, fontweight='bold')
        plt.xlabel('策略')
        plt.ylabel('夏普比率')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
        
        for i, (bar, ratio) in enumerate(zip(bars, sharpe_ratios)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{ratio:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 4. 最大回撤對比
        plt.subplot(2, 3, 4)
        max_drawdowns = []
        for results in all_results.values():
            rewards = results['reward'][0] if len(results['reward'].shape) > 1 else results['reward']
            cumulative_returns = np.cumsum(rewards)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            max_drawdowns.append(np.max(drawdown))
        
        bars = plt.bar(range(len(strategy_names)), max_drawdowns, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('最大回撤對比', fontsize=14, fontweight='bold')
        plt.xlabel('策略')
        plt.ylabel('最大回撤')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
        
        for i, (bar, dd) in enumerate(zip(bars, max_drawdowns)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{dd:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 5. 交易頻率對比
        plt.subplot(2, 3, 5)
        trading_frequencies = []
        for results in all_results.values():
            actions = results['action'][0] if len(results['action'].shape) > 1 else results['action']
            action_changes = np.sum(np.diff(actions) != 0)
            freq = action_changes / len(actions)
            trading_frequencies.append(freq)
        
        bars = plt.bar(range(len(strategy_names)), trading_frequencies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('交易頻率對比', fontsize=14, fontweight='bold')
        plt.xlabel('策略')
        plt.ylabel('交易頻率')
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=45, ha='right')
        
        for i, (bar, freq) in enumerate(zip(bars, trading_frequencies)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{freq:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 6. 風險收益散點圖
        plt.subplot(2, 3, 6)
        volatilities = []
        for results in all_results.values():
            rewards = results['reward'][0] if len(results['reward'].shape) > 1 else results['reward']
            volatilities.append(np.std(rewards))
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        for i, (strategy_name, vol, ret) in enumerate(zip(strategy_names, volatilities, return_rates)):
            plt.scatter(vol, ret, s=100, c=colors[i], label=strategy_name, alpha=0.7, edgecolors='black')
            plt.annotate(strategy_name, (vol, ret), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('風險-收益散點圖', fontsize=14, fontweight='bold')
        plt.xlabel('波動率（風險）')
        plt.ylabel('收益率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"性能比較圖已保存至: {os.path.join(save_path, 'strategy_comparison.png')}")
    
    def generate_report(self, save_path="./analysis"):
        """生成完整的分析報告"""
        print("="*60)
        print("           MacroHFT 動態混合策略性能分析報告")
        print("="*60)
        
        # 載入和比較所有策略
        all_results, all_metrics = self.compare_strategies()
        
        if not all_metrics:
            print("❌ 無可用的策略結果進行分析")
            return
        
        # 創建比較表
        comparison_df = self.create_comparison_table(all_metrics)
        
        if comparison_df is not None:
            print("\n📊 策略性能比較表:")
            print("-" * 80)
            print(comparison_df.to_string(index=False))
            
            # 保存比較表
            os.makedirs(save_path, exist_ok=True)
            comparison_df.to_csv(os.path.join(save_path, 'strategy_comparison.csv'), index=False, encoding='utf-8-sig')
            print(f"\n💾 比較表已保存至: {os.path.join(save_path, 'strategy_comparison.csv')}")
        
        # 生成性能分析
        print("\n📈 性能分析:")
        print("-" * 40)
        
        # 找出最佳策略
        best_return_strategy = max(all_metrics.items(), key=lambda x: x[1]['total_return'] if x[1] else 0)
        best_sharpe_strategy = max(all_metrics.items(), key=lambda x: x[1]['sharpe_ratio'] if x[1] else 0)
        best_drawdown_strategy = min(all_metrics.items(), key=lambda x: x[1]['max_drawdown'] if x[1] else float('inf'))
        
        print(f"🏆 最高收益率策略: {best_return_strategy[0]} ({best_return_strategy[1]['total_return']:.4f})")
        print(f"📊 最佳夏普比率策略: {best_sharpe_strategy[0]} ({best_sharpe_strategy[1]['sharpe_ratio']:.4f})")
        print(f"🛡️  最小回撤策略: {best_drawdown_strategy[0]} ({best_drawdown_strategy[1]['max_drawdown']:.4f})")
        
        # 繪製比較圖
        self.plot_performance_comparison(all_results, save_path)
        
        # 策略優劣分析
        print(f"\n🔍 策略分析總結:")
        print("-" * 40)
        
        for strategy_name, metrics in all_metrics.items():
            if metrics:
                print(f"\n{strategy_name}:")
                print(f"  • 收益率: {metrics['total_return']:.4f}")
                print(f"  • 夏普比率: {metrics['sharpe_ratio']:.4f}")
                print(f"  • 最大回撤: {metrics['max_drawdown']:.4f}")
                print(f"  • 交易頻率: {metrics['trading_frequency']:.4f}")
                print(f"  • 勝率: {metrics['win_rate']:.4f}")
        
        print(f"\n✅ 完整分析報告已生成並保存至: {save_path}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='MacroHFT策略性能評估工具')
    parser.add_argument('--result_path', type=str, default='./result/high_level/ETHUSDT', 
                        help='結果文件夾路徑')
    parser.add_argument('--save_path', type=str, default='./analysis', 
                        help='分析結果保存路徑')
    
    args = parser.parse_args()
    
    evaluator = StrategyEvaluator(args.result_path)
    evaluator.generate_report(args.save_path)


if __name__ == "__main__":
    main()