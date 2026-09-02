import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_performance import StrategyPerformance

def process_trading_data(df):
    # Initialize variables
    trades = []
    current_position = None
    position_size = 0
    entry_price = 0
    entry_date = None
    initial_capital = 10000
    current_capital = initial_capital
    
    # Process each trade
    for idx, row in df.iterrows():
        date = row['date']
        action = row['action']
        price = row['price']
        shares = row['shares']
        
        if action == 'buy' and shares > 0 and current_position is None:
            # Open new position
            current_position = 'long'
            position_size = current_capital/price
            entry_price = price
            entry_date = date
        
        elif action == 'sell' and current_position == 'long':
            # Close position
            exit_price = price
            exit_date = date
            
            # Calculate profit
            profit = (exit_price - entry_price) * position_size
            profit_pct = (exit_price - entry_price) / entry_price * 100
            holding_period = (exit_date - entry_date).days if entry_date is not None else 1
            
            # Assuming transaction cost is 0.1%
            fee = position_size * entry_price * 0.001 + position_size * exit_price * 0.001
            net_profit = profit - fee
            
            # Record trade
            trades.append({
                'entry_time': entry_date,
                'exit_time': exit_date,
                'position': 'long',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'net_profit': net_profit,
                'net_pnl_pct': profit_pct,
                'total_fee': fee,
                'holding_periods': holding_period
            })
            
            # Update capital
            current_capital += net_profit
            
            # Reset position
            current_position = None
            position_size = 0
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Create equity curve (daily asset changes)
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    equity_data = []
    
    current_equity = initial_capital
    for date in date_range:
        # Check if there are completed trades
        if not trades_df.empty:
            completed_trades = trades_df[trades_df['exit_time'] <= date]
            if not completed_trades.empty:
                current_equity = initial_capital + completed_trades['net_profit'].sum()
        
        equity_data.append({
            'time': date,
            'equity': current_equity
        })
    
    equity_df = pd.DataFrame(equity_data)
    
    # Calculate drawdown
    equity_df['running_max'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['running_max'] - equity_df['equity']) / equity_df['running_max']
    
    drawdown_df = equity_df[['time', 'drawdown']].copy()
    
    # Prepare price data
    price_df = df[['date', 'price']].copy()
    price_df.columns = ['open_time', 'close']
    price_df['open'] = price_df['close']
    price_df['high'] = price_df['close'] * 1.005 # default 
    price_df['low'] = price_df['close'] * 0.995 # default
    
    return trades_df, equity_df, drawdown_df, price_df

if __name__ == "__main__":

    df = pd.read_csv('datasets/trade_records.csv')
    df['date'] = pd.to_datetime(df['date'])

    # processing dataframe
    trades_df, equity_df, drawdown_df, price_data = process_trading_data(df)

    print("trades_df：")
    print(trades_df)
    print("\nequity_df：")
    print(equity_df.head())
    print("\ndrawdown_df：")
    print(drawdown_df)
    print("\nprice_data：")
    print(price_data.head())

    # Create StrategyPerformance object and analyze
    perf = StrategyPerformance(trades_df, equity_df, price_data, drawdown_df, 10000)

    # Calculate performance metrics
    perf.calculate_metrics()

    # Print results
    perf.print_metrics()

    # Generate report (make sure the directory exists)
    output_dir = 'performance_report'
    perf.generate_report(output_dir)

    print(f"\nPerformance report generated in {output_dir} directory!")