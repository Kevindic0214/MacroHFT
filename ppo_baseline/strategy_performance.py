import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from matplotlib.dates import date2num

class StrategyPerformance:
    """
    A class to calculate performance metrics and generate visualizations for trading strategies
    """
    
    def __init__(self, trades_df=None, equity_df=None, price_df=None, drawdown_df=None, initial_capital=10000):
        """
        Initialize the performance calculator
        
        Parameters:
        - trades_df: DataFrame containing trade details
        - equity_df: DataFrame containing equity curve data
        - drawdown_df: DataFrame containing drawdown data
        - initial_capital: Initial capital for the strategy
        - price_df: DataFrame containing price data
        """
        self.trades_df = trades_df
        self.equity_df = equity_df
        self.drawdown_df = drawdown_df
        self.initial_capital = initial_capital
        self.price_data = price_df
        self.metrics = {}
    
    def set_data(self, trades_df, equity_df, drawdown_df, price_df):
        """Set data for performance calculation"""
        self.trades_df = trades_df
        self.equity_df = equity_df
        self.drawdown_df = drawdown_df
        self.price_data = price_df
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.trades_df is None or self.equity_df is None or self.drawdown_df is None:
            raise ValueError("Trade data, equity data, and drawdown data must be set before calculating metrics")
        
        # Basic trade metrics
        self.metrics['total_trades'] = len(self.trades_df)
        self.metrics['profitable_trades'] = len(self.trades_df[self.trades_df['net_profit'] > 0])
        self.metrics['losing_trades'] = len(self.trades_df[self.trades_df['net_profit'] <= 0])
        self.metrics['win_rate'] = self.metrics['profitable_trades'] / self.metrics['total_trades'] if self.metrics['total_trades'] > 0 else 0
        
        # PnL metrics
        self.metrics['total_profit'] = self.trades_df['net_profit'].sum()
        self.metrics['total_return'] = self.metrics['total_profit'] / self.initial_capital
        
        # Calculate returns
        self.equity_df['return'] = self.equity_df['equity'].pct_change()
        
        # Time-based metrics
        start_date = self.equity_df['time'].min()
        end_date = self.equity_df['time'].max()
        days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
        self.metrics['trading_days'] = days
        self.metrics['years'] = days / 250
        
        # Annualized return
        self.metrics['annualized_return'] = (1 + self.metrics['total_return']) ** (1 / self.metrics['years']) - 1 if self.metrics['years'] > 0 else 0
        
        # Risk metrics
        returns = self.equity_df['return'].dropna()
        
        # Sharpe ratio (assuming risk-free rate of 0)
        self.metrics['daily_returns_mean'] = returns.mean()
        self.metrics['daily_returns_std'] = returns.std()
        
        # Annualization factor
        annualization_factor = np.sqrt(250)
        
        self.metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * annualization_factor if returns.std() > 0 else 0
        
        # Sortino ratio (downside risk only)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std()
        self.metrics['sortino_ratio'] = (returns.mean() / downside_deviation) * annualization_factor if downside_deviation > 0 else 0
        
        # Maximum drawdown
        self.metrics['max_drawdown'] = self.drawdown_df['drawdown'].max()
        
        # Maximum drawdown duration
        if self.metrics['max_drawdown'] > 0:
            # Find the peak before the max drawdown
            max_dd_idx = self.drawdown_df['drawdown'].idxmax()
            max_dd_time = self.drawdown_df.loc[max_dd_idx, 'time']
            
            # Find the last time equity was at the peak before max drawdown
            peak_equity = self.equity_df['equity'].iloc[:max_dd_idx].max()
            peak_time_idx = self.equity_df['equity'].iloc[:max_dd_idx].idxmax()
            peak_time = self.equity_df.loc[peak_time_idx, 'time']
            
            # Find the first time equity recovered to the peak after max drawdown
            recovery_mask = (self.equity_df['time'] > max_dd_time) & (self.equity_df['equity'] >= peak_equity)
            
            if recovery_mask.any():
                recovery_time_idx = recovery_mask.idxmax()
                recovery_time = self.equity_df.loc[recovery_time_idx, 'time']
                self.metrics['max_drawdown_duration'] = (recovery_time - peak_time).total_seconds() / (24 * 60 * 60)  # in days
            else:
                # If no recovery, drawdown duration is from peak to end of backtest
                self.metrics['max_drawdown_duration'] = (end_date - peak_time).total_seconds() / (24 * 60 * 60)  # in days
        else:
            self.metrics['max_drawdown_duration'] = 0
        
        # Calmar ratio
        self.metrics['calmar_ratio'] = self.metrics['annualized_return'] / self.metrics['max_drawdown'] if self.metrics['max_drawdown'] > 0 else float('inf')
        
        # Average trade metrics
        self.metrics['avg_profit_per_trade'] = self.trades_df['net_profit'].mean()
        self.metrics['avg_profit_pct_per_trade'] = self.trades_df['net_pnl_pct'].mean()
        self.metrics['avg_win'] = self.trades_df.loc[self.trades_df['net_profit'] > 0, 'net_profit'].mean() if self.metrics['profitable_trades'] > 0 else 0
        self.metrics['avg_loss'] = self.trades_df.loc[self.trades_df['net_profit'] <= 0, 'net_profit'].mean() if self.metrics['losing_trades'] > 0 else 0
        self.metrics['win_loss_ratio'] = abs(self.metrics['avg_win'] / self.metrics['avg_loss']) if self.metrics['avg_loss'] != 0 else float('inf')
        
        # Profit factor
        gross_profit = self.trades_df.loc[self.trades_df['net_profit'] > 0, 'net_profit'].sum()
        gross_loss = abs(self.trades_df.loc[self.trades_df['net_profit'] <= 0, 'net_profit'].sum())
        self.metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average holding period
        self.metrics['avg_holding_periods'] = self.trades_df['holding_periods'].mean()
        
        # Total fees
        self.metrics['total_fees'] = self.trades_df['total_fee'].sum()
        
        # Expectancy
        self.metrics['expectancy'] = (self.metrics['win_rate'] * self.metrics['avg_win']) + \
                                    ((1 - self.metrics['win_rate']) * self.metrics['avg_loss'])
        
        # System quality number
        self.metrics['sqn'] = (self.metrics['expectancy'] / self.trades_df['net_profit'].std()) * \
                             np.sqrt(self.metrics['total_trades']) if self.trades_df['net_profit'].std() > 0 else 0
        
        return self.metrics
    
    def plot_equity_curve(self, output_path=None, show_plot=True):
        """
        Plot equity curve
        
        Parameters:
        - output_path: Path to save the plot
        - show_plot: Whether to display the plot
        """
        if self.equity_df is None:
            raise ValueError("Equity data must be set before plotting")
        
        plt.figure(figsize=(14, 7))
        
        # Plot equity curve
        plt.plot(self.equity_df['time'], self.equity_df['equity'], label='Equity', color='blue')
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Format y-axis to show currency
        def currency_formatter(x, pos):
            return f'${x:,.0f}'
        
        plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        # Add labels and title
        plt.title('Equity Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Equity curve saved to {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_drawdown_curve(self, output_path=None, show_plot=True):
        """
        Plot drawdown curve
        
        Parameters:
        - output_path: Path to save the plot
        - show_plot: Whether to display the plot
        """
        if self.drawdown_df is None:
            raise ValueError("Drawdown data must be set before plotting")
        
        plt.figure(figsize=(14, 7))
        
        # Plot drawdown curve
        plt.plot(self.drawdown_df['time'], self.drawdown_df['drawdown'] * 100, color='red')
        
        # Fill area under the curve
        plt.fill_between(self.drawdown_df['time'], self.drawdown_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Add labels and title
        plt.title('Drawdown Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Drawdown curve saved to {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_trade_pnl(self, output_path=None, show_plot=True):
        """
        Plot individual trade PnL
        
        Parameters:
        - output_path: Path to save the plot
        - show_plot: Whether to display the plot
        """
        if self.trades_df is None:
            raise ValueError("Trade data must be set before plotting")
        
        plt.figure(figsize=(14, 7))
        
        # Create trade indices
        trade_indices = range(len(self.trades_df))
        
        # Get PnL values
        pnl_values = self.trades_df['net_profit'].values
        
        # Create colors based on profit/loss
        colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
        
        # Plot bars
        plt.bar(trade_indices, pnl_values, color=colors)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.title('Individual Trade PnL', fontsize=16)
        plt.xlabel('Trade Number', fontsize=12)
        plt.ylabel('Profit/Loss ($)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Trade PnL plot saved to {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_cumulative_returns(self, output_path=None, show_plot=True):
        """
        Plot cumulative returns
        
        Parameters:
        - output_path: Path to save the plot
        - show_plot: Whether to display the plot
        """
        if self.equity_df is None:
            raise ValueError("Equity data must be set before plotting")
        
        plt.figure(figsize=(14, 7))
        
        # Calculate cumulative returns
        initial_equity = self.equity_df['equity'].iloc[0]
        cumulative_returns = (self.equity_df['equity'] / initial_equity - 1) * 100
        
        # Plot cumulative returns
        plt.plot(self.equity_df['time'], cumulative_returns, label='Cumulative Return', color='green')
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # Add labels and title
        plt.title('Cumulative Returns', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Cumulative returns plot saved to {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_monthly_returns(self, output_path=None, show_plot=True):
        """
        Plot monthly returns heatmap
        
        Parameters:
        - output_path: Path to save the plot
        - show_plot: Whether to display the plot
        """
        if self.equity_df is None:
            raise ValueError("Equity data must be set before plotting")
        
        # Resample equity curve to get monthly returns
        self.equity_df['time'] = pd.to_datetime(self.equity_df['time'], utc=True)
        self.equity_df['month'] = self.equity_df['time'].dt.to_period('M')
        monthly_returns = self.equity_df.groupby('month')['equity'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
        )
        
        # Convert to DataFrame with year and month columns
        monthly_returns_df = pd.DataFrame(monthly_returns).reset_index()
        monthly_returns_df['year'] = monthly_returns_df['month'].dt.year
        monthly_returns_df['month_num'] = monthly_returns_df['month'].dt.month
        
        # Pivot the data for heatmap
        pivot_table = monthly_returns_df.pivot_table(
            index='year', columns='month_num', values='equity'
        )
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        
        # Define colormap with red for negative and green for positive
        cmap = sns.diverging_palette(10, 120, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt=".2f", 
            cmap=cmap,
            center=0, 
            linewidths=.5, 
            cbar_kws={"label": "Monthly Return (%)"}
        )
        
        # Set month names as column labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12) + 0.5, month_names)
        
        # Add labels and title
        plt.title('Monthly Returns (%)', fontsize=16)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path)
            print(f"Monthly returns heatmap saved to {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, output_dir, show_plots=False):
        """
        Generate all plots and save them to the output directory
        
        Parameters:
        - output_dir: Directory to save the plots
        - show_plots: Whether to display the plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        self.plot_equity_curve(os.path.join(output_dir, 'equity_curve.png'), show_plots)
        self.plot_drawdown_curve(os.path.join(output_dir, 'drawdown_curve.png'), show_plots)
        self.plot_trade_pnl(os.path.join(output_dir, 'trade_pnl.png'), show_plots)
        self.plot_cumulative_returns(os.path.join(output_dir, 'cumulative_returns.png'), show_plots)
        self.plot_monthly_returns(os.path.join(output_dir, 'monthly_returns.png'), show_plots)
        
        # Plot price chart with entry/exit points if price data is available
        if hasattr(self, 'price_data') and self.price_data is not None:
            self.plot_price_with_trades(os.path.join(output_dir, 'price_with_trades.png'), show_plots)
    
    def set_price_data(self, price_data):
        """
        Set price data for plotting price chart with entry/exit points
        
        Parameters:
        - price_data: DataFrame containing price data with columns: time, open, high, low, close
        """
        self.price_data = price_data
    
    def plot_price_with_trades(self, output_path=None, show_plot=True):
        """
        Plot price chart with entry/exit points
        
        Parameters:
        - output_path: Path to save the plot
        - show_plot: Whether to display the plot
        """
        if not hasattr(self, 'price_data') or self.price_data is None or self.trades_df is None or self.trades_df.empty:
            print("Price data and trade data are required for plotting")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot price line
        plt.plot(self.price_data['open_time'], self.price_data['close'], color='black', linewidth=1, label='Price')
        
        # Get trade date range
        min_date = self.trades_df['entry_time'].min() - pd.Timedelta(days=5)
        max_date = self.trades_df['exit_time'].max() + pd.Timedelta(days=5)
        
        # Limit chart range
        plt.xlim(min_date, max_date)
        
        # Mark trade points
        for _, trade in self.trades_df.iterrows():
            # Get trade information
            entry_time = trade['entry_time']
            entry_price = trade['entry_price']
            position = trade['position']
            exit_time = trade['exit_time']
            exit_price = trade['exit_price']
            
            # Entry point
            marker = '^' if position == 'long' else 'v'
            color = 'green' if position == 'long' else 'red'
            plt.scatter(entry_time, entry_price, color=color, marker=marker, s=100, zorder=5)
            
            # Exit point
            exit_color = 'red' if position == 'long' else 'green'
            plt.scatter(exit_time, exit_price, color=exit_color, marker='x', s=100, zorder=5)
            
            # Connect entry and exit points
            plt.plot([entry_time, exit_time], [entry_price, exit_price], 
                    color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Add profit/loss label
            profit = exit_price - entry_price if position == 'long' else entry_price - exit_price
            profit_percent = profit / entry_price * 100
            profit_text = f"{profit_percent:.2f}%"
            
            # Add profit/loss label at the midpoint of the line
            mid_time = entry_time + (exit_time - entry_time) / 2
            mid_price = (entry_price + exit_price) / 2
            
            plt.annotate(profit_text, 
                        xy=(mid_time, mid_price),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', 
                                fc='green' if profit > 0 else 'red', 
                                alpha=0.3))
        
        # Add legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='black', lw=1),
            Line2D([0], [0], marker='^', color='green', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='v', color='red', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='x', color='black', markersize=8, linestyle='None')
        ]
        plt.legend(custom_lines, ['Price', 'Long Entry', 'Short Entry', 'Exit'], loc='best')
        
        # Add grid and title
        plt.grid(True, alpha=0.3)
        plt.title('Trade Entry and Exit Points', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price')
        
        # Format dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save or display
        if output_path:
            plt.savefig(output_path)
            print(f"Trade chart saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, output_dir, show_plots=False):
        """
        Generate all plots and save them to the output directory
        
        Parameters:
        - output_dir: Directory to save the plots
        - show_plots: Whether to display the plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        self.plot_equity_curve(os.path.join(output_dir, 'equity_curve.png'), show_plots)
        self.plot_drawdown_curve(os.path.join(output_dir, 'drawdown_curve.png'), show_plots)
        self.plot_trade_pnl(os.path.join(output_dir, 'trade_pnl.png'), show_plots)
        self.plot_cumulative_returns(os.path.join(output_dir, 'cumulative_returns.png'), show_plots)
        self.plot_monthly_returns(os.path.join(output_dir, 'monthly_returns.png'), show_plots)
        
        # Plot price chart with entry/exit points if price data is available
        if hasattr(self, 'price_data') and self.price_data is not None:
            self.plot_price_with_trades(os.path.join(output_dir, 'price_with_trades.png'), show_plots)
    
    def set_price_data(self, price_data):
        """
        Set price data for plotting price chart with entry/exit points
        
        Parameters:
        - price_data: DataFrame containing price data with columns: time, open, high, low, close
        """
        self.price_data = price_data
    
    def print_metrics(self):
        """Print performance metrics in a formatted way"""
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n===== PERFORMANCE METRICS =====")
        print(f"Total Trades: {self.metrics['total_trades']}")
        print(f"Win Rate: {self.metrics['win_rate']:.2%}")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"Expectancy: ${self.metrics['expectancy']:.2f}")
        print(f"System Quality Number (SQN): {self.metrics['sqn']:.2f}")
        print("\n--- Returns ---")
        print(f"Total Return: {self.metrics['total_return']:.2%}")
        print(f"Annualized Return: {self.metrics['annualized_return']:.2%}")
        print("\n--- Risk Metrics ---")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2%}")
        print(f"Maximum Drawdown Duration: {self.metrics['max_drawdown_duration']:.1f} days")
        print(f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")
        print("\n--- Trade Statistics ---")
        print(f"Average Profit per Trade: ${self.metrics['avg_profit_per_trade']:.2f} ({self.metrics['avg_profit_pct_per_trade']:.2%})")
        print(f"Average Win: ${self.metrics['avg_win']:.2f}")
        print(f"Average Loss: ${self.metrics['avg_loss']:.2f}")
        print(f"Win/Loss Ratio: {self.metrics['win_loss_ratio']:.2f}")
        print(f"Average Holding Period: {self.metrics['avg_holding_periods']:.2f} candles")
        print(f"Total Fees: ${self.metrics['total_fees']:.2f}")
        print(f"Expectancy: ${self.metrics['expectancy']:.2f}")
        print("\n================================")
    
    def generate_html_report(self, output_path):
        """Generate HTML performance report"""
        if not self.metrics:
            self.calculate_metrics()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-value {{ font-weight: bold; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Trading Strategy Performance Report</h1>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Trades</td><td>{self.metrics['total_trades']}</td></tr>
                    <tr><td>Win Rate</td><td>{self.metrics['win_rate']:.2%}</td></tr>
                    <tr><td>Profit Factor</td><td>{self.metrics['profit_factor']:.2f}</td></tr>
                    <tr><td>Total Return</td><td class="{('positive' if self.metrics['total_return'] >= 0 else 'negative')}">{self.metrics['total_return']:.2%}</td></tr>
                    <tr><td>Annualized Return</td><td class="{('positive' if self.metrics['annualized_return'] >= 0 else 'negative')}">{self.metrics['annualized_return']:.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{self.metrics['sharpe_ratio']:.2f}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{self.metrics['sortino_ratio']:.2f}</td></tr>
                    <tr><td>Maximum Drawdown</td><td class="negative">{self.metrics['max_drawdown']:.2%}</td></tr>
                    <tr><td>Maximum Drawdown Duration</td><td>{self.metrics['max_drawdown_duration']:.1f} days</td></tr>
                    <tr><td>Calmar Ratio</td><td>{self.metrics['calmar_ratio']:.2f}</td></tr>
                    <tr><td>System Quality Number (SQN)</td><td>{self.metrics['sqn']:.2f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Trade Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Profitable Trades</td><td>{self.metrics['profitable_trades']} ({self.metrics['win_rate']:.2%})</td></tr>
                    <tr><td>Losing Trades</td><td>{self.metrics['losing_trades']} ({1-self.metrics['win_rate']:.2%})</td></tr>
                    <tr><td>Average Profit per Trade</td><td class="{('positive' if self.metrics['avg_profit_per_trade'] >= 0 else 'negative')}">${self.metrics['avg_profit_per_trade']:.2f} ({self.metrics['avg_profit_pct_per_trade']:.2%})</td></tr>
                    <tr><td>Average Win</td><td class="positive">${self.metrics['avg_win']:.2f}</td></tr>
                    <tr><td>Average Loss</td><td class="negative">${self.metrics['avg_loss']:.2f}</td></tr>
                    <tr><td>Win/Loss Ratio</td><td>{self.metrics['win_loss_ratio']:.2f}</td></tr>
                    <tr><td>Expectancy</td><td class="{('positive' if self.metrics['expectancy'] >= 0 else 'negative')}">${self.metrics['expectancy']:.2f}</td></tr>
                    <tr><td>Average Holding Period</td><td>{self.metrics['avg_holding_periods']:.2f} candles</td></tr>
                    <tr><td>Total Fees</td><td>${self.metrics['total_fees']:.2f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Equity Curve</h2>
                <img src="equity_curve.png" alt="Equity Curve">
            </div>
            
            <div class="section">
                <h2>Cumulative Returns</h2>
                <img src="cumulative_returns.png" alt="Cumulative Returns">
            </div>
            
            <div class="section">
                <h2>Drawdown Curve</h2>
                <img src="drawdown_curve.png" alt="Drawdown Curve">
            </div>
            
            <div class="section">
                <h2>Individual Trade PnL</h2>
                <img src="trade_pnl.png" alt="Individual Trade PnL">
            </div>
            
            <div class="section">
                <h2>Monthly Returns Heatmap</h2>
                <img src="monthly_returns.png" alt="Monthly Returns Heatmap">
            </div>
            
            <div class="section">
                <h2>Price Chart with Trades</h2>
                <img src="price_with_trades.png" alt="Price Chart with Trades">
            </div>
        </body>
        </html>
        """
        
        # Write HTML content to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report generated at {output_path}")
    
    def generate_report(self, output_dir):
        """
        Generate comprehensive performance report
        
        Parameters:
        - output_dir: Directory to save the report and plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics if not already calculated
        if not self.metrics:
            self.calculate_metrics()
        
        # Generate all plots
        self.plot_all(output_dir, show_plots=False)
        
        # Generate HTML report
        report_path = os.path.join(output_dir, 'performance_report.html')
        self.generate_html_report(report_path)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([self.metrics])
        metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"Performance report and metrics generated in {output_dir}")


if __name__ == "__main__":
    # This is just a test to ensure the class works correctly
    # In practice, this class would be imported and used by the strategy backtest script
    
    # Create sample data
    trades_df = pd.DataFrame({
        'entry_time': pd.date_range(start='2021-01-01', periods=10, freq='D'),
        'exit_time': pd.date_range(start='2021-01-02', periods=10, freq='D'),
        'position': ['long', 'short'] * 5,
        'entry_price': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
        'exit_price': [105, 100, 115, 110, 125, 120, 135, 130, 145, 140],
        'net_profit': [5, -5, 5, -5, 5, -5, 5, -5, 5, -5],
        'net_pnl_pct': [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05],
        'total_fee': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'holding_periods': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    
    equity_df = pd.DataFrame({
        'time': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'equity': [10000 + i * 100 for i in range(100)]
    })
    
    drawdown_df = pd.DataFrame({
        'time': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'drawdown': [0.01 * (i % 10) / 10 for i in range(100)]
    })
    
    price_data = pd.DataFrame({
        'open_time': pd.date_range(start='2021-01-01', periods=100, freq='D'),
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [100 + i * 0.1 + 1 for i in range(100)],
        'low': [100 + i * 0.1 - 1 for i in range(100)],
        'close': [100 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })
    
    # Create performance calculator
    perf = StrategyPerformance(trades_df, equity_df, price_data, drawdown_df, 10000)
    
    # Calculate metrics
    perf.calculate_metrics()
    
    # Print metrics
    perf.print_metrics()
    
    # Generate report
    perf.generate_report('test_performance_report')
