import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns
from tabulate import tabulate
from strategy_performance import StrategyPerformance
import pytz

class ATRTrendStrategy:
    """
    ATR Trend Strategy Backtester for Stock Market
    
    Strategy Logic:
    1. Calculate ATR with a period of 14
    2. Determine trend direction based on ATR
    3. Enter long when trend changes from down to up (executed on next day's open)
    4. Enter short when trend changes from up to down (executed on next day's open)
    5. Exit positions based on take profit (9x ATR), stop loss (3x ATR), or trend reversal (executed on next day's open)
    """
    
    def __init__(self, data_path, atr_period=14, threshold_multiplier=3, tp_multiplier=9, 
                 sl_multiplier=3, fee_rate=0.0005, initial_capital=10000, 
                 start_date=None, end_date=None, long_only=False, obv_filter=False, obv_period=5, obv_multiplier=1.0):
        """
        Initialize the strategy
        
        Parameters:
        - data_path: Path to the CSV file with price data
        - atr_period: Period for ATR calculation (default: 14)
        - threshold_multiplier: Multiplier for ATR to determine trend change (default: 3)
        - tp_multiplier: Multiplier for ATR to set take profit (default: 9)
        - sl_multiplier: Multiplier for ATR to set stop loss (default: 3)
        - fee_rate: Trading fee rate (default: 0.0005 = 5bp)
        - initial_capital: Initial capital for backtesting (default: 10000)
        - start_date: Start date for backtesting (format: 'YYYY-MM-DD')
        - end_date: End date for backtesting (format: 'YYYY-MM-DD')
        - long_only: If True, only take long positions (default: False)
        - obv_filter: If True, use OBV as an additional filter for entries (default: False)
        - obv_period: Period for OBV moving average calculation (default: 5)
        - obv_multiplier: Multiplier for OBV filter - requires OBV > multiplier * OBV_MA for long (default: 1.0)
        """
        self.data_path = data_path
        self.atr_period = atr_period
        self.threshold_multiplier = threshold_multiplier
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.fee_rate = fee_rate
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.long_only = long_only
        self.obv_filter = obv_filter
        self.obv_period = obv_period
        self.obv_multiplier = obv_multiplier
        
        # Load and prepare data
        self.df = self.load_data()
        self.df = self.df.reset_index(drop=True)
        self.prepare_data()
        
        # Initialize trade tracking
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Initialize performance calculator
        self.performance = None
    
    def load_data(self):
        """Load price data from CSV file and filter by date range if specified"""
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp columns to datetime
        df['open_time'] = pd.to_datetime(df['open_time'])
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_time'])
        
        # Filter by date range if specified
        if self.start_date:
            start_date = pd.to_datetime(self.start_date)
            start_date = pytz.timezone('Asia/Taipei').localize(start_date)
            df = df[df['open_time'] >= start_date]
        
        if self.end_date:
            end_date = pd.to_datetime(self.end_date)
            end_date = pytz.timezone('Asia/Taipei').localize(end_date)
            df = df[df['open_time'] <= end_date]
        
        # Convert price and volume columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Convert optional numeric columns if they exist
        optional_numeric_columns = ['quote_asset_volume', 'taker_buy_base_asset_volume',
                                   'taker_buy_quote_asset_volume', 'number_of_trades']
        for col in optional_numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        return df
    
    def prepare_data(self):
        """Prepare data for backtesting by calculating ATR and trend direction"""
        # Calculate True Range
        self.df['tr'] = self.calculate_true_range()
        
        # Calculate ATR
        self.df['atr'] = self.df['tr'].rolling(window=self.atr_period).mean()
        
        # Calculate OBV and its moving average if OBV filter is enabled
        if self.obv_filter:
            self.calculate_obv()
        
        # Calculate trend direction
        self.calculate_trend_direction()
        
        # Add signal columns for next-day execution
        self.df['signal'] = 'none'
        self.df['next_day_action'] = 'none'
        
        # Generate signals
        self.generate_signals()
        
        # Drop rows with NaN values (due to ATR calculation)
        self.df = self.df.dropna().reset_index(drop=True)
    
    def calculate_true_range(self):
        """Calculate True Range for ATR"""
        high_low = self.df['high'] - self.df['low']
        high_close_prev = abs(self.df['high'] - self.df['close'].shift(1))
        low_close_prev = abs(self.df['low'] - self.df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr
    
    def calculate_obv(self):
        """Calculate On-Balance Volume (OBV) and its moving average"""
        # Initialize OBV
        obv = [0]
        
        # Calculate OBV
        for i in range(1, len(self.df)):
            if self.df.loc[i, 'close'] > self.df.loc[i-1, 'close']:
                obv.append(obv[-1] + self.df.loc[i, 'volume'])
            elif self.df.loc[i, 'close'] < self.df.loc[i-1, 'close']:
                obv.append(obv[-1] - self.df.loc[i, 'volume'])
            else:
                obv.append(obv[-1])
        
        # Add OBV to dataframe
        self.df['obv'] = obv
        
        # Calculate OBV moving average
        self.df['obv_ma'] = self.df['obv'].rolling(window=self.obv_period).mean()
        
        # Create OBV signal (1 for OBV > multiplier*OBV_MA, -1 for OBV < multiplier*OBV_MA, 0 for equal)
        self.df['obv_signal'] = np.where(self.df['obv'] > self.obv_multiplier * self.df['obv_ma'], 1, 
                                        np.where(self.df['obv'] < self.df['obv_ma'] / self.obv_multiplier, -1, 0))
    
    def calculate_trend_direction(self):
        """Calculate trend direction based on ATR threshold"""
        # Initialize with up trend
        up_trend = True
        last_high = self.df['high'].iloc[0]
        last_low = self.df['low'].iloc[0]
        directions = []
        
        for i in range(len(self.df)):
            # Skip rows with NaN ATR values
            if pd.isna(self.df.loc[i, 'atr']):
                directions.append(None)
                continue
                
            threshold = self.threshold_multiplier * self.df.loc[i, 'atr']
            
            if up_trend:
                if self.df.loc[i, 'high'] > last_high:
                    last_high = self.df.loc[i, 'high']
                elif self.df.loc[i, 'close'] < (last_high - threshold):
                    up_trend = False
                    last_low = self.df.loc[i, 'low']
            else:
                if self.df.loc[i, 'low'] < last_low:
                    last_low = self.df.loc[i, 'low']
                elif self.df.loc[i, 'close'] > (last_low + threshold):
                    up_trend = True
                    last_high = self.df.loc[i, 'high']
                    
            directions.append('up' if up_trend else 'down')
        
        self.df['direction'] = directions
    
    def generate_signals(self):
        """Generate trading signals for next-day execution"""
        # Skip the first row since we need previous direction
        for i in range(1, len(self.df)):
            current_direction = self.df.loc[i, 'direction']
            previous_direction = self.df.loc[i-1, 'direction']
            
            # Check for trend change
            if current_direction != previous_direction:
                if current_direction == 'up':
                    # Check OBV filter for long entry if enabled
                    obv_condition = True
                    if self.obv_filter and 'obv_signal' in self.df.columns:
                        obv_signal = self.df.loc[i, 'obv_signal']
                        obv_condition = (obv_signal == 1)  # OBV > multiplier * OBV_MA
                    
                    if obv_condition:
                        self.df.loc[i, 'signal'] = 'buy'
                elif current_direction == 'down' and not self.long_only:
                    # Check OBV filter for short entry if enabled
                    obv_condition = True
                    if self.obv_filter and 'obv_signal' in self.df.columns:
                        obv_signal = self.df.loc[i, 'obv_signal']
                        obv_condition = (obv_signal == -1)  # OBV < OBV_MA for short entry
                    
                    if obv_condition:
                        self.df.loc[i, 'signal'] = 'sell'
        
        # Set next_day_action based on the previous day's signal
        # This will be used to execute trades on the next day's open
        for i in range(1, len(self.df)):
            self.df.loc[i, 'next_day_action'] = self.df.loc[i-1, 'signal']
    
    def run_backtest(self):
        """Run the backtest with next-day execution for stock market"""
        # Initialize variables
        position = None  # 'long', 'short', or None
        entry_price = 0
        entry_time = None
        entry_index = 0
        atr_at_entry = 0
        capital = self.initial_capital
        take_profit_price = None
        stop_loss_price = None
        
        # Loop through each day
        for i in range(1, len(self.df)):
            current_time = self.df.loc[i, 'open_time']
            current_open = self.df.loc[i, 'open']
            current_high = self.df.loc[i, 'high']
            current_low = self.df.loc[i, 'low']
            current_close = self.df.loc[i, 'close']
            current_atr = self.df.loc[i, 'atr']
            current_direction = self.df.loc[i, 'direction']
            next_day_action = self.df.loc[i, 'next_day_action']
            
            # Execute any pending actions at the open price
            if position is None and next_day_action in ['buy', 'sell']:
                # Execute the buy at the open price
                if next_day_action == 'buy':
                    position = 'long'
                    entry_price = current_open
                    entry_time = current_time
                    entry_index = i
                    atr_at_entry = current_atr
                    
                    # Set take profit and stop loss levels
                    take_profit_price = entry_price + (self.tp_multiplier * atr_at_entry)
                    stop_loss_price = entry_price - (self.sl_multiplier * atr_at_entry)
                    
                elif next_day_action == 'sell' and not self.long_only:
                    position = 'short'
                    entry_price = current_open
                    entry_time = current_time
                    entry_index = i
                    atr_at_entry = current_atr
                    
                    # Set take profit and stop loss levels
                    take_profit_price = entry_price - (self.tp_multiplier * atr_at_entry)
                    stop_loss_price = entry_price + (self.sl_multiplier * atr_at_entry)
            
            # Check for exit conditions if we have a position
            elif position is not None:
                exit_triggered = False
                exit_price = None
                exit_reason = None
                
                # Check if trend has changed and we need to exit on the next day
                if (position == 'long' and current_direction == 'down') or \
                   (position == 'short' and current_direction == 'up'):
                    # Set signal to exit and update next_day_action for the next day
                    self.df.loc[i, 'signal'] = 'exit'
                    # Make sure we're not at the last index before trying to set i+1
                    if i + 1 < len(self.df):
                        self.df.loc[i+1, 'next_day_action'] = 'exit'
                
                # Check for take profit or stop loss during the day
                if position == 'long':
                    # Take profit hit during the day
                    if current_high >= take_profit_price:
                        # Check if current open price is higher than take profit price
                        if current_open >= take_profit_price:
                            # Use current open price
                            exit_price = current_open
                        else:
                            # Use take profit price
                            exit_price = take_profit_price
                        exit_reason = 'take_profit'
                        exit_triggered = True
                    # Stop loss hit during the day
                    elif current_low <= stop_loss_price:
                        # Check if current open price is lower than stop loss price
                        if current_open <= stop_loss_price:
                            # Use current open price
                            exit_price = current_open
                        else:
                            # Use stop loss price
                            exit_price = stop_loss_price
                        exit_reason = 'stop_loss'
                        exit_triggered = True
                        
                elif position == 'short':
                    # Take profit hit during the day
                    if current_low <= take_profit_price:
                        # Check if current open price is lower than take profit price
                        if current_open <= take_profit_price:
                            # Use current open price
                            exit_price = current_open
                        else:
                            # Use take profit price
                            exit_price = take_profit_price
                        exit_reason = 'take_profit'
                        exit_triggered = True
                    # Stop loss hit during the day
                    elif current_high >= stop_loss_price:
                        # Check if current open price is higher than stop loss price
                        if current_open >= stop_loss_price:
                            # Use current open price
                            exit_price = current_open
                        else:
                            # Use stop loss price
                            exit_price = stop_loss_price
                        exit_reason = 'stop_loss'
                        exit_triggered = True
                
                # Execute exit if triggered
                if exit_triggered:
                    # Close position
                    self.close_position(position, entry_time, current_time, entry_price, exit_price, 
                                       entry_index, i, atr_at_entry, exit_reason, capital)
                    
                    # Update capital
                    if position == 'long':
                        pnl = (exit_price - entry_price) / entry_price
                    else:  # short
                        pnl = (entry_price - exit_price) / entry_price
                    
                    capital *= (1 + pnl - 2 * self.fee_rate)  # Subtract fees for entry and exit
                    
                    # Reset position
                    position = None
                    take_profit_price = None
                    stop_loss_price = None
                
                # Check if we need to exit based on previous day's signal
                elif next_day_action == 'exit':
                    exit_price = current_open
                    exit_reason = 'trend_reversal'
                    
                    # Close position
                    self.close_position(position, entry_time, current_time, entry_price, exit_price, 
                                      entry_index, i, atr_at_entry, exit_reason, capital)
                    
                    # Update capital
                    if position == 'long':
                        pnl = (exit_price - entry_price) / entry_price
                    else:  # short
                        pnl = (entry_price - exit_price) / entry_price
                    
                    capital *= (1 + pnl - 2 * self.fee_rate)  # Subtract fees for entry and exit
                    
                    # Check if we need to open a new position in the opposite direction
                    if position == 'long' and not self.long_only:
                        # Open new short position
                        obv_condition = True
                        if self.obv_filter and 'obv_signal' in self.df.columns:
                            obv_signal = self.df.loc[i-1, 'obv_signal']
                            obv_condition = (obv_signal == -1)
                        
                        if obv_condition:
                            position = 'short'
                            entry_price = current_open
                            entry_time = current_time
                            entry_index = i
                            atr_at_entry = current_atr
                            
                            # Set take profit and stop loss levels
                            take_profit_price = entry_price - (self.tp_multiplier * atr_at_entry)
                            stop_loss_price = entry_price + (self.sl_multiplier * atr_at_entry)
                        else:
                            position = None
                            take_profit_price = None
                            stop_loss_price = None
                    
                    elif position == 'short':
                        # Open new long position
                        obv_condition = True
                        if self.obv_filter and 'obv_signal' in self.df.columns:
                            obv_signal = self.df.loc[i-1, 'obv_signal']
                            obv_condition = (obv_signal == 1)
                        
                        if obv_condition:
                            position = 'long'
                            entry_price = current_open
                            entry_time = current_time
                            entry_index = i
                            atr_at_entry = current_atr
                            
                            # Set take profit and stop loss levels
                            take_profit_price = entry_price + (self.tp_multiplier * atr_at_entry)
                            stop_loss_price = entry_price - (self.sl_multiplier * atr_at_entry)
                        else:
                            position = None
                            take_profit_price = None
                            stop_loss_price = None

                    else:
                        position = None
                        take_profit_price = None
                        stop_loss_price = None
            
            # Calculate current equity including unrealized P&L
            current_equity = capital
            
            # If we have an open position, add unrealized P&L to equity
            if position is not None:
                # Calculate unrealized P&L based on current close price
                if position == 'long':
                    unrealized_pnl = (current_close - entry_price) / entry_price
                else:  # short
                    unrealized_pnl = (entry_price - current_close) / entry_price
                
                # Add unrealized P&L to equity (without considering fees yet since position is not closed)
                current_equity = capital * (1 + unrealized_pnl)
            
            # Update equity curve with current equity (including unrealized P&L)
            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity
            })

            # Record position each day
            self.record_positions(position, current_time, current_equity)
            
            # Calculate drawdown based on current equity
            if len(self.equity_curve) > 1:  
                peak_equity = max([e['equity'] for e in self.equity_curve[:-1]])
                peak_equity = max(peak_equity, current_equity)  
            else:
                peak_equity = current_equity  

            # Calculate current drawdown
            current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0

            # Append to drawdown curve
            self.drawdown_curve.append({
                'time': current_time,
                'drawdown': current_drawdown
            })

        # Close any open position at the end of the backtest
        if position is not None:
            last_index = len(self.df) - 1
            last_time = self.df.loc[last_index, 'open_time']
            last_price = self.df.loc[last_index, 'close']
            
            exit_reason = 'end_of_backtest'
            self.close_position(position, entry_time, last_time, entry_price, last_price, 
                               entry_index, last_index, atr_at_entry, exit_reason, capital)
            
            # Update capital
            if position == 'long':
                pnl = (last_price - entry_price) / entry_price
            else:  # short
                pnl = (entry_price - last_price) / entry_price
                
            capital *= (1 + pnl - 2 * self.fee_rate)  # Subtract fees for entry and exit
        
        # Convert trades and equity curve to DataFrames
        self.trades_df = pd.DataFrame(self.trades)
        self.equity_df = pd.DataFrame(self.equity_curve)
        self.drawdown_df = pd.DataFrame(self.drawdown_curve)
        self.positions_df = pd.DataFrame(self.positions)
        
        # Initialize performance calculator
        self.performance = StrategyPerformance(
            trades_df=self.trades_df,
            equity_df=self.equity_df,
            price_df = self.df,
            drawdown_df=self.drawdown_df,
            initial_capital=self.initial_capital
        )
        
        # Calculate performance metrics
        self.performance.calculate_metrics()
    
    def record_positions(self, position, current_time, capital):
        """Record long short position with daily return"""
        
        # Calculate daily return
        daily_return = 0
        if len(self.positions) > 0:
            previous_capital = self.positions[-1]['capital']
            if previous_capital > 0:  # Avoid division by zero
                daily_return = (capital - previous_capital) / previous_capital
        
        # Record position
        self.positions.append({
            'time': current_time,
            'position': position,
            'capital': capital,
            'daily_return': daily_return
        })

    def close_position(self, position, entry_time, exit_time, entry_price, exit_price, 
                      entry_index, exit_index, atr_at_entry, exit_reason, capital):
        """Record a closed position"""
        # Calculate PnL
        if position == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) / entry_price
        
        # Calculate fees
        entry_fee = entry_price * self.fee_rate
        exit_fee = exit_price * self.fee_rate
        total_fee = entry_fee + exit_fee
        
        # Calculate net PnL
        net_pnl = pnl - (total_fee / entry_price)
        
        # Calculate profit in currency
        profit = capital * pnl
        net_profit = capital * net_pnl
        
        # Record the trade
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'atr_at_entry': atr_at_entry,
            'exit_reason': exit_reason,
            'holding_periods': exit_index - entry_index,
            'pnl_pct': pnl * 100,  # Convert to percentage
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'total_fee': total_fee,
            'net_pnl_pct': net_pnl * 100,  # Convert to percentage
            'profit': profit,
            'net_profit': net_profit,
            'capital_after': capital * (1 + net_pnl)
        }
        
        self.trades.append(trade)

    def save_positions(self, output_path):
        """Save position data to CSV"""
        if not self.positions:
            print("No positions to save.")
            return
    
        # map position to numeric
        position_map = {
            'long': 1,
            'short': -1,
            None: 0
        } 
        self.positions_df['position'] = self.positions_df['position'].map(position_map)

        # Save positions to CSV
        self.positions_df.to_csv(output_path, index=False)

        
    def save_trade_details(self, output_path):
        """Save trade details to CSV"""
        if not self.trades:
            print("No trades to save.")
            return
        
        # Select relevant columns for trade details
        trade_details = self.trades_df[[
            'entry_time', 'exit_time', 'position', 'entry_price', 'exit_price',
            'total_fee', 'net_pnl_pct', 'net_profit', 'exit_reason'
        ]]
        
        # Save to CSV
        trade_details.to_csv(output_path, index=False)
        print(f"Trade details saved to {output_path}")
    
    def generate_report(self, output_dir):
        """Generate comprehensive backtest report"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trade details
        trade_details_path = os.path.join(output_dir, 'trade_details.csv')
        self.save_trade_details(trade_details_path)

        # Save position details
        position_details_path = os.path.join(output_dir, 'position_details.csv')
        self.save_positions(position_details_path)
        
        # Generate performance report
        self.performance.generate_report(output_dir)
        
        print(f"Backtest report generated in {output_dir}")


def main():
    """Main function to run the backtest"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run ATR Trend Strategy Backtest')
    parser.add_argument('--data_path', type=str, default=os.path.join('data', '^IXIC_1d_data_20250504_001.csv'), 
                        help='Path to the CSV file with price data')
    parser.add_argument('--start_date', type=str, default='2021-01-01', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--atr_period', type=int, default=14, help='ATR period')
    parser.add_argument('--threshold', type=float, default=3, help='Threshold multiplier for trend change')
    parser.add_argument('--tp', type=float, default=6, help='Take profit multiplier')
    parser.add_argument('--sl', type=float, default=2, help='Stop loss multiplier')
    parser.add_argument('--fee', type=float, default=0.0002565, help='Fee rate (e.g., 0.0005 for 5bp)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--long_only', action='store_true', help='Only take long positions')
    parser.add_argument('--obv_filter', action='store_true', help='Use OBV as an additional filter for entries')
    parser.add_argument('--obv_period', type=int, default=10, help='Period for OBV moving average calculation')
    parser.add_argument('--obv_multiplier', type=float, default=1, help='Multiplier for OBV filter - higher values make the filter more strict')
    
    args = parser.parse_args()

    # Define paths
    data_path = os.path.join('data', '^IXIC_1d_data_20250504_001.csv')
    output_dir = os.path.join('results', 'atr_trend_strategy', f'backtest_{args.start_date}_{args.end_date}')

    # Create strategy instance
    strategy = ATRTrendStrategy(
        data_path=data_path,
        atr_period=args.atr_period,
        threshold_multiplier=args.threshold,
        tp_multiplier=args.tp,
        sl_multiplier=args.sl,
        fee_rate=args.fee,
        initial_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
        long_only=args.long_only,
        obv_filter=args.obv_filter,
        obv_period=args.obv_period,
        obv_multiplier=args.obv_multiplier
    )
    
    # Run backtest
    strategy.run_backtest()
    
    # Print performance summary
    strategy.performance.print_metrics()

    # Generate report
    strategy.generate_report(output_dir)


if __name__ == "__main__":
    main()
