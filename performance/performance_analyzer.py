import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class NumPyPerformanceAnalyzer:
    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.metrics = {}
        self.data = {}
        
        # Load all .npy files
        self.load_numpy_data()
        
    def load_numpy_data(self):
        """Load all .npy files from the directory"""
        try:
            # Load all the .npy files
            files_to_load = [
                'action.npy',
                'reward.npy', 
                'commission_fee_history.npy',
                'require_money.npy',
                'final_balance.npy'
            ]
            
            for file_name in files_to_load:
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    data = np.load(file_path, allow_pickle=True)
                    key = file_name.replace('.npy', '')
                    self.data[key] = data
                    print(f"Loaded {file_name}: shape {data.shape}")
                else:
                    print(f"File not found: {file_name}")
            
            # Process the data
            self.process_loaded_data()
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def process_loaded_data(self):
        """Process the loaded data into usable formats"""
        
        # Extract scalar values
        if 'commission_fee_history' in self.data:
            self.total_fees = float(self.data['commission_fee_history'].item() if self.data['commission_fee_history'].size == 1 
                                   else self.data['commission_fee_history'].sum())
        
        if 'require_money' in self.data:
            self.required_money = float(self.data['require_money'].item() if self.data['require_money'].size == 1
                                       else self.data['require_money'].sum())
        
        if 'final_balance' in self.data:
            self.final_balance = float(self.data['final_balance'].item() if self.data['final_balance'].size == 1
                                      else self.data['final_balance'][-1])
        
        # Process time series data
        if 'action' in self.data:
            # Flatten if 2D
            self.actions = self.data['action'].flatten() if self.data['action'].ndim > 1 else self.data['action']
            
        if 'reward' in self.data:
            # Flatten if 2D  
            self.rewards = self.data['reward'].flatten() if self.data['reward'].ndim > 1 else self.data['reward']
        
        self.n_periods = len(self.actions) if hasattr(self, 'actions') else len(self.rewards)
        self.time_index = np.arange(self.n_periods)
        # Original setting doesnt have initial capital
        self.initial_capital = self.required_money
        
        print(f"Processed data:")
        print(f"   - Total periods: {self.n_periods}")
        print(f"   - Total fees: ${self.total_fees:.2f}")
        print(f"   - Required money: ${self.required_money:.2f}")
        print(f"   - Final balance: ${self.final_balance:.2f}")
    
    def identify_trades(self):
        if not hasattr(self, 'actions'):
            return []
        
        trades = []
        in_position = False
        entry_idx = None
        
        for i, action in enumerate(self.actions):
            if action == 1 and not in_position:  # Enter position
                in_position = True
                entry_idx = i
                
            elif action == 0 and in_position:  # Exit position
                if entry_idx is not None:
                    # Calculate trade reward (sum of rewards during holding period)
                    trade_reward = self.rewards[entry_idx:i+1].sum() if hasattr(self, 'rewards') else 0
                    
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_time': self.time_index[entry_idx],
                        'exit_time': self.time_index[i],
                        'holding_periods': i - entry_idx + 1,
                        'trade_reward': trade_reward,
                        'trade_return': trade_reward / self.initial_capital if self.initial_capital > 0 else 0
                    })
                
                in_position = False
                entry_idx = None
        
        # Handle case where last trade is still open
        if in_position and entry_idx is not None:
            trade_reward = self.rewards[entry_idx:].sum() if hasattr(self, 'rewards') else 0
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': len(self.actions) - 1,
                'entry_time': self.time_index[entry_idx],
                'exit_time': self.time_index[-1],
                'holding_periods': len(self.actions) - entry_idx,
                'trade_reward': trade_reward,
                'trade_return': trade_reward / self.initial_capital if self.initial_capital > 0 else 0
            })
        
        return trades
    
    def calculate_equity_curve(self):
        if not hasattr(self, 'rewards'):
            return pd.DataFrame()
        
        # Calculate cumulative returns
        cumulative_rewards = np.cumsum(self.rewards)
        equity_values = self.initial_capital + cumulative_rewards
        
        equity_df = pd.DataFrame({
            'time': self.time_index,
            'equity': equity_values,
            'minute_return': self.rewards / self.initial_capital,
            'cumulative_return': cumulative_rewards / self.initial_capital
        })
        
        return equity_df
    
    def calculate_drawdown(self, equity_df):
        if equity_df.empty:
            return pd.DataFrame()
        
        # Make a copy and ensure all values are numeric
        drawdown_df = equity_df.copy()
        
        # Convert columns to numeric, coercing errors to NaN
        drawdown_df['equity'] = pd.to_numeric(drawdown_df['equity'], errors='coerce')
        
        # Drop any rows with NaN values in equity
        drawdown_df = drawdown_df.dropna(subset=['equity'])
        
        if drawdown_df.empty:
            print("Warning: No valid equity data for drawdown calculation")
            return pd.DataFrame()
        
        # Calculate peak and drawdown
        drawdown_df['peak'] = drawdown_df['equity'].expanding().max()
        
        # Calculate drawdown
        drawdown_df['drawdown'] = (drawdown_df['equity'] - drawdown_df['peak']) / drawdown_df['peak']
        
        return drawdown_df[['time', 'drawdown', 'peak']]
    
    def calculate_metrics(self): 
        # Get trades
        trades = self.identify_trades()
        
        # Get equity curve
        equity_df = self.calculate_equity_curve()
        
        # Get drawdown
        drawdown_df = self.calculate_drawdown(equity_df) if not equity_df.empty else pd.DataFrame()
        
        # Basic trade metrics
        self.metrics['total_trades'] = len(trades)
        
        if trades:
            profitable_trades = [t for t in trades if t['trade_reward'] > 0]
            losing_trades = [t for t in trades if t['trade_reward'] <= 0]
            
            self.metrics['profitable_trades'] = len(profitable_trades)
            self.metrics['losing_trades'] = len(losing_trades)
            self.metrics['win_rate'] = len(profitable_trades) / len(trades) if trades else 0
            
            # Trade statistics
            trade_rewards = [t['trade_reward'] for t in trades]
            self.metrics['avg_profit_per_trade'] = np.mean(trade_rewards) if trade_rewards else 0
            self.metrics['avg_profit_pct_per_trade'] = np.mean([t['trade_return'] for t in trades]) if trades else 0
            
            if profitable_trades:
                self.metrics['avg_win'] = np.mean([t['trade_reward'] for t in profitable_trades])
            else:
                self.metrics['avg_win'] = 0
                
            if losing_trades:
                self.metrics['avg_loss'] = np.mean([t['trade_reward'] for t in losing_trades])
            else:
                self.metrics['avg_loss'] = 0
            
            # Win/Loss ratio
            if self.metrics['avg_loss'] != 0:
                self.metrics['win_loss_ratio'] = abs(self.metrics['avg_win'] / self.metrics['avg_loss'])
            else:
                self.metrics['win_loss_ratio'] = float('inf') if self.metrics['avg_win'] > 0 else 0
            
            # Profit factor
            gross_profit = sum([t['trade_reward'] for t in profitable_trades])
            gross_loss = abs(sum([t['trade_reward'] for t in losing_trades]))
            self.metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average holding periods
            self.metrics['avg_holding_periods'] = np.mean([t['holding_periods'] for t in trades])
            
            # Expectancy
            self.metrics['expectancy'] = (self.metrics['win_rate'] * self.metrics['avg_win']) + \
                                        ((1 - self.metrics['win_rate']) * self.metrics['avg_loss'])
            
            # System Quality Number (SQN)
            if len(trade_rewards) > 1:
                trade_std = np.std(trade_rewards)
                if trade_std > 0:
                    self.metrics['sqn'] = (self.metrics['expectancy'] / trade_std) * np.sqrt(len(trades))
                else:
                    self.metrics['sqn'] = 0
            else:
                self.metrics['sqn'] = 0
        
        else:
            # No trades identified
            self.metrics.update({
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit_per_trade': 0,
                'avg_profit_pct_per_trade': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'win_loss_ratio': 0,
                'profit_factor': 0,
                'avg_holding_periods': 0,
                'expectancy': 0,
                'sqn': 0
            })
        
        # Overall performance metrics
        if hasattr(self, 'final_balance') and hasattr(self, 'initial_capital'):
            self.metrics['total_profit'] = self.final_balance 
            self.metrics['total_return'] = self.metrics['total_profit'] / self.initial_capital
        else:
            self.metrics['total_profit'] = self.rewards.sum() if hasattr(self, 'rewards') else 0
            self.metrics['total_return'] = self.metrics['total_profit'] / self.initial_capital
        
        # Time-based metrics for minute-level data in 24/7 cryptocurrency markets
        self.metrics['trading_minutes'] = self.n_periods
        # Convert minutes to days (for crypto: 24 hours = 1440 minutes per day)
        self.metrics['trading_days'] = self.n_periods / 1440
        # Convert days to years (for crypto: 365 days per year, as markets operate 24/7)
        self.metrics['years'] = self.metrics['trading_days'] / 365
        
        # Annualized return
        if self.metrics['years'] > 0:
            self.metrics['annualized_return'] = (1 + self.metrics['total_return']) ** (1 / self.metrics['years']) - 1
        else:
            self.metrics['annualized_return'] = 0
        
        # Risk metrics (from equity curve)
        if not equity_df.empty and 'minute_return' in equity_df.columns:
            returns = equity_df['minute_return'].dropna()
            
            if len(returns) > 1:
                self.metrics['minute_returns_mean'] = returns.mean()
                self.metrics['minute_returns_std'] = returns.std()
                
                # Sharpe ratio (annualized) - adjusted for minute data in 24/7 crypto markets
                if self.metrics['minute_returns_std'] > 0:
                    # For crypto minute data: sqrt(365 days * 1440 minutes per day)
                    self.metrics['sharpe_ratio'] = (self.metrics['minute_returns_mean'] / self.metrics['minute_returns_std']) * np.sqrt(365 * 1440)
                else:
                    self.metrics['sharpe_ratio'] = 0
                
                # Sortino ratio
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_std = negative_returns.std()
                    if downside_std > 0:
                        # For crypto minute data: sqrt(365 days * 1440 minutes per day)
                        self.metrics['sortino_ratio'] = (self.metrics['minute_returns_mean'] / downside_std) * np.sqrt(365 * 1440)
                    else:
                        self.metrics['sortino_ratio'] = 0
                else:
                    self.metrics['sortino_ratio'] = float('inf') if self.metrics['minute_returns_mean'] > 0 else 0
            else:
                self.metrics.update({
                    'minute_returns_mean': 0,
                    'minute_returns_std': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0
                })
        
        # Drawdown metrics
        if not drawdown_df.empty:
            self.metrics['max_drawdown'] = abs(drawdown_df['drawdown'].min())
            
            # Calculate max drawdown duration (simplified)
            if self.metrics['max_drawdown'] > 0:
                # Find periods in drawdown
                in_drawdown = drawdown_df['drawdown'] < 0
                if in_drawdown.any():
                    # Simple approach: find longest consecutive drawdown period
                    drawdown_periods = []
                    current_period = 0
                    for is_dd in in_drawdown:
                        if is_dd:
                            current_period += 1
                        else:
                            if current_period > 0:
                                drawdown_periods.append(current_period)
                            current_period = 0
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    
                    self.metrics['max_drawdown_duration'] = max(drawdown_periods) if drawdown_periods else 0
                else:
                    self.metrics['max_drawdown_duration'] = 0
            else:
                self.metrics['max_drawdown_duration'] = 0
                
            # Calmar ratio
            if self.metrics['max_drawdown'] > 0:
                self.metrics['calmar_ratio'] = self.metrics['annualized_return'] / self.metrics['max_drawdown']
            else:
                self.metrics['calmar_ratio'] = float('inf') if self.metrics['annualized_return'] > 0 else 0
        else:
            self.metrics.update({
                'max_drawdown': 0,
                'max_drawdown_duration': 0,
                'calmar_ratio': float('inf') if self.metrics.get('annualized_return', 0) > 0 else 0
            })
        
        # Fees
        self.metrics['total_fees'] = getattr(self, 'total_fees', 0)
        
        return self.metrics
    
    def print_metrics(self):
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "="*60)
        print("TRADING PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nTRADE STATISTICS")
        print("-" * 40)
        print(f"Total Trades:              {self.metrics['total_trades']}")
        print(f"Profitable Trades:         {self.metrics['profitable_trades']}")
        print(f"Losing Trades:             {self.metrics['losing_trades']}")
        print(f"Win Rate:                  {self.metrics['win_rate']:.2%}")
        print(f"Average Holding Periods:   {self.metrics['avg_holding_periods']:.2f}")
        
        print(f"\nRETURN METRICS")
        print("-" * 40)
        print(f"Total Profit:              ${self.metrics['total_profit']:.2f}")
        print(f"Total Return:              {self.metrics['total_return']:.2%}")
        print(f"Annualized Return:         {self.metrics['annualized_return']:.2%}")
        print(f"Average Profit per Trade:  ${self.metrics['avg_profit_per_trade']:.2f}")
        print(f"Average Profit % per Trade: {self.metrics['avg_profit_pct_per_trade']:.2%}")
        
        print(f"\nRISK METRICS")
        print("-" * 40)
        print(f"Minute Returns Mean:        {self.metrics['minute_returns_mean']:.6f}")
        print(f"Minute Returns Std:         {self.metrics['minute_returns_std']:.6f}")
        print(f"Sharpe Ratio:              {self.metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio:             {self.metrics['sortino_ratio']:.4f}")
        print(f"Maximum Drawdown:          {self.metrics['max_drawdown']:.2%}")
        print(f"Max Drawdown Duration:     {self.metrics['max_drawdown_duration']:.0f} days")
        print(f"Calmar Ratio:              {self.metrics['calmar_ratio']:.4f}")
        
        print(f"\nTRADE QUALITY METRICS")
        print("-" * 40)
        print(f"Average Win:               ${self.metrics['avg_win']:.2f}")
        print(f"Average Loss:              ${self.metrics['avg_loss']:.2f}")
        print(f"Win/Loss Ratio:            {self.metrics['win_loss_ratio']:.4f}")
        print(f"Profit Factor:             {self.metrics['profit_factor']:.4f}")
        print(f"Expectancy:                ${self.metrics['expectancy']:.2f}")
        print(f"System Quality Number:     {self.metrics['sqn']:.4f}")
        
        print(f"\nCOST METRICS")
        print("-" * 40)
        print(f"Total Fees:                ${self.metrics['total_fees']:.2f}")
        
        print(f"\nTIME METRICS")
        print("-" * 40)
        print(f"Trading Days:              {self.metrics['trading_days']:.0f}")
        print(f"Trading Years:             {self.metrics['years']:.2f}")
        
        print("="*60)
    
    def plot_performance(self, output_dir=None, show_plots=True):
        """Generate performance plots"""
        
        equity_df = self.calculate_equity_curve()
        
        if equity_df.empty:
            print("Cannot generate plots: No equity data available")
            return
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. Equity Curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['time'], equity_df['equity'], 'b-', linewidth=2)
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # 2. Cumulative Returns
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['time'], equity_df['cumulative_return'] * 100, 'g-', linewidth=2)
        plt.title('Cumulative Returns (%)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # 3. Drawdown
        drawdown_df = self.calculate_drawdown(equity_df)
        if not drawdown_df.empty:
            try:
                plt.figure(figsize=(12, 6))
                
                # Convert data to numeric and handle any potential issues
                time_values = drawdown_df['time'].values
                drawdown_values = pd.to_numeric(drawdown_df['drawdown'], errors='coerce').fillna(0).values * 100
                
                # Check for any invalid values before plotting
                if np.isfinite(drawdown_values).all():
                    plt.fill_between(time_values, drawdown_values, 0, color='red', alpha=0.3)
                    plt.plot(time_values, drawdown_values, 'r-', linewidth=1)
                    plt.title('Drawdown (%)')
                    plt.xlabel('Date')
                    plt.ylabel('Drawdown (%)')
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    if output_dir:
                        plt.savefig(os.path.join(output_dir, 'drawdown.png'))
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                else:
                    print("Warning: Cannot plot drawdown due to invalid values")
                    plt.close()
            except Exception as e:
                print(f"Error plotting drawdown: {e}")
                plt.close()
        
        # 4. Trade PnL
        trades = self.identify_trades()
        if trades:
            plt.figure(figsize=(14, 7))
            
            # Create trade indices
            trade_indices = range(1, len(trades) + 1)
            
            # Get PnL values
            pnl_values = [trade['trade_reward'] for trade in trades]
            
            # Create colors based on profit/loss
            colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
            
            # Plot bars
            bars = plt.bar(trade_indices, pnl_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add labels and title
            plt.title('Individual Trade PnL', fontsize=16)
            plt.xlabel('Trade Number', fontsize=12)
            plt.ylabel('Profit/Loss ($)', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text box
            profitable_trades = len([p for p in pnl_values if p > 0])
            losing_trades = len([p for p in pnl_values if p <= 0])
            avg_profit = np.mean([p for p in pnl_values if p > 0]) if profitable_trades > 0 else 0
            avg_loss = np.mean([p for p in pnl_values if p <= 0]) if losing_trades > 0 else 0
            
            stats_text = f'Total Trades: {len(trades)}\n'
            stats_text += f'Profitable: {profitable_trades} ({profitable_trades/len(trades):.1%})\n'
            stats_text += f'Losing: {losing_trades} ({losing_trades/len(trades):.1%})\n'
            stats_text += f'Avg Win: ${avg_profit:.2f}\n'
            stats_text += f'Avg Loss: ${avg_loss:.2f}'
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Format x-axis for better readability
            if len(trades) > 20:
                # Show every nth trade number on x-axis
                step = max(1, len(trades) // 10)
                plt.xticks(range(1, len(trades) + 1, step))
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'trade_pnl.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()
        else:
            print("No trades identified, skipping Trade PnL plot")
        
        # 5. Cumulative Trade PnL
        if trades:
            plt.figure(figsize=(14, 7))
            
            # Calculate cumulative PnL
            cumulative_pnl = np.cumsum([trade['trade_reward'] for trade in trades])
            trade_indices = range(1, len(trades) + 1)
            
            # Plot cumulative PnL
            try:
                # Ensure data is numeric and finite
                trade_indices = np.array(trade_indices)
                cumulative_pnl = np.array(cumulative_pnl, dtype=float)
                
                # Check for any invalid values
                if np.isfinite(cumulative_pnl).all():
                    plt.plot(trade_indices, cumulative_pnl, 'b-', linewidth=2, marker='o', markersize=4)
                    
                    # Color the area based on profit/loss
                    plt.fill_between(trade_indices, cumulative_pnl, 0, 
                                   where=(cumulative_pnl >= 0), color='green', alpha=0.3, label='Profit')
                    plt.fill_between(trade_indices, cumulative_pnl, 0, 
                               where=(cumulative_pnl < 0), color='red', alpha=0.3, label='Loss')
                
                    # Add horizontal line at y=0
                    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    plt.title('Cumulative Trade PnL', fontsize=16)
                    plt.xlabel('Trade Number', fontsize=12)
                    plt.ylabel('Cumulative Profit/Loss ($)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Format x-axis
                    if len(trades) > 20:
                        step = max(1, len(trades) // 10)
                        plt.xticks(range(1, len(trades) + 1, step))
                    
                    plt.tight_layout()
                    
                    if output_dir:
                        plt.savefig(os.path.join(output_dir, 'cumulative_trade_pnl.png'))
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                else:
                    print("Warning: Cannot plot cumulative PnL due to invalid values")
                    plt.close()
            except Exception as e:
                print(f"Error plotting cumulative trade PnL: {e}")
                plt.close()
        
        # 6. Minute Returns Distribution
        if 'minute_return' in equity_df.columns:
            try:
                plt.figure(figsize=(10, 6))
                
                # Convert to numeric and drop NaN values
                returns = pd.to_numeric(equity_df['minute_return'], errors='coerce').dropna()
                
                if len(returns) > 0:
                    # Calculate statistics
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Create histogram
                    n, bins, patches = plt.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
                    
                    # Color bars based on positive/negative
                    for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
                        if bin_center < 0:
                            patch.set_facecolor('red')
                        else:
                            patch.set_facecolor('green')
                    
                    # Add mean line
                    plt.axvline(mean_return, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.6f}')
                    
                    # Add standard deviation lines
                    plt.axvline(mean_return + std_return, color='purple', linestyle=':', alpha=0.7, label=f'+1σ: {mean_return + std_return:.6f}')
                    plt.axvline(mean_return - std_return, color='purple', linestyle=':', alpha=0.7, label=f'-1σ: {mean_return - std_return:.6f}')
                    
                    plt.title('Minute Returns Distribution')
                    plt.xlabel('Minute Return')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    if output_dir:
                        plt.savefig(os.path.join(output_dir, 'minute_returns_distribution.png'))
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
                else:
                    print("Warning: No valid minute return data for distribution plot")
                    plt.close()
            except Exception as e:
                print(f"Error plotting minute returns distribution: {e}")
                plt.close()
    
    def save_metrics_to_csv(self, output_path):
        """Save metrics to CSV file"""
        if not self.metrics:
            self.calculate_metrics()
        
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")

# Usage example
def analyze_numpy_results(data_dir , output_dir=None):
    print(f"Starting analysis of: {data_dir}")
    
    # Create analyzer
    analyzer = NumPyPerformanceAnalyzer(data_dir)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Print results
    analyzer.print_metrics()
    
    # Generate plots
    if output_dir:
        analyzer.plot_performance(output_dir, show_plots=False)
        analyzer.save_metrics_to_csv(os.path.join(output_dir, 'calculated_metrics.csv'))
        print(f"Results saved to {output_dir}")
    
    return analyzer, metrics

def combine_multiple_results(data_dirs, labels=None, output_dir=None, show_plots=True):
    if labels is None:
        labels = [os.path.basename(data_dir) for data_dir in data_dirs]
    
    if len(labels) != len(data_dirs):
        raise ValueError("Labels length must match data_dirs length")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    analyzers = []
    equity_dfs = []
    drawdown_dfs = []
    
    # Create analyzer for each folder and get data
    print("Loading and processing data from multiple directories...")
    for i, data_dir in enumerate(data_dirs):
        try:
            print(f"Processing {labels[i]}: {data_dir}")
            analyzer = NumPyPerformanceAnalyzer(data_dir)
            analyzer.calculate_metrics()
            
            equity_df = analyzer.calculate_equity_curve()
            if not equity_df.empty:
                equity_df.loc[:, 'label'] = labels[i]
                equity_dfs.append(equity_df)
                
                drawdown_df = analyzer.calculate_drawdown(equity_df)
                if not drawdown_df.empty:
                    drawdown_df.loc[:, 'label'] = labels[i]
                    drawdown_dfs.append(drawdown_df)
                
                analyzers.append((analyzer, labels[i]))
                print(f"Successfully processed {labels[i]}")
            else:
                print(f"No equity data found for {labels[i]}")
                
        except Exception as e:
            print(f"Error processing {labels[i]}: {e}")
    
    if not equity_dfs:
        print("No valid data found in any directory!")
        return
    
    # Generate color list
    colors = plt.cm.tab10(np.linspace(0, 1, len(equity_dfs)))
    
    # 1. Combined Cumulative Returns Chart
    plt.figure(figsize=(14, 8))
    
    for i, equity_df in enumerate(equity_dfs):
        plt.plot(equity_df['time'], equity_df['cumulative_return'] * 100, 
                linewidth=2.5, label=equity_df['label'].iloc[0], color=colors[i])
    
    plt.title('Combined Cumulative Returns Comparison (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    
    # Add statistics text box
    stats_text = "Final Returns:\n"
    for i, (analyzer, label) in enumerate(analyzers):
        final_return = analyzer.metrics.get('total_return', 0) * 100
        stats_text += f"{label}: {final_return:.2f}%\n"
    
    plt.text(0.02, 0.98, stats_text.strip(), transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'combined_cumulative_returns.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'combined_cumulative_returns.png')}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Combined Drawdown Chart
    if drawdown_dfs:
        plt.figure(figsize=(14, 8))
        
        for i, drawdown_df in enumerate(drawdown_dfs):
            plt.fill_between(drawdown_df['time'], drawdown_df['drawdown'] * 100, 0, 
                           color=colors[i], alpha=0.3)
            plt.plot(drawdown_df['time'], drawdown_df['drawdown'] * 100, 
                    linewidth=2, label=drawdown_df['label'].iloc[0], color=colors[i])
        
        plt.title('Combined Drawdown Comparison (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        
        # Add max drawdown statistics
        dd_stats_text = "Max Drawdown:\n"
        for i, (analyzer, label) in enumerate(analyzers):
            max_dd = analyzer.metrics.get('max_drawdown', 0) * 100
            dd_stats_text += f"{label}: {max_dd:.2f}%\n"
        
        plt.text(0.02, 0.02, dd_stats_text.strip(), transform=plt.gca().transAxes, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'combined_drawdown.png'), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(output_dir, 'combined_drawdown.png')}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

# Usage example function
def plot_multiple_strategies():
    # Define directories to compare
    data_directories = [
        "dynamic_result",
        "macrohft_result", 
        "rainbowdqn_result",
        "qrdqn_result",
        "multipatchformer_result"
    ]
    
    # Define labels (optional)
    strategy_labels = [
        "Dynamic",
        "MacroHFT",
        "RainbowDQN",
        "QRDQN",
        "MultiPatchFormer",
    ]
    
    # Execute combined analysis
    try:
        combine_multiple_results(
            data_dirs=data_directories,
            labels=strategy_labels,
            output_dir="combined_performance_analysis",
            show_plots=True
        )
        
        print(f"Results saved to: combined_performance_analysis/")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    # Example usage
    data_directory_list = [
        "dynamic_result",
        "macrohft_result", 
        "rainbowdqn_result",
        "qrdqn_result",
        "multipatchformer_result"
    ]
    
    # Run analysis
    for data_directory in data_directory_list:
        analyzer, metrics = analyze_numpy_results(
            data_directory,
            output_dir=f"performance_analysis_output/{data_directory}"
        )

    # Execute multi-strategy analysis
    plot_multiple_strategies()