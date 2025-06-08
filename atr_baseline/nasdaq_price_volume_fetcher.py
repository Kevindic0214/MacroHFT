import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys
import io
import yfinance as yf
from FinMind.data import DataLoader

class NasdaqIndexFetcher:
    """
    A class to fetch historical price and volume data for Nasdaq index
    from Yahoo Finance and save it to CSV files.
    """
    
    def __init__(self, data_folder="data", max_file_size_mb=50, api_token=""):
        # Symbol for Nasdaq Composite Index
        self.symbol = "^IXIC"
        
        # Interval (daily)
        self.interval = "1d"
        
        # Data storage settings
        self.data_folder = data_folder
        self.max_file_size_mb = max_file_size_mb
        
        # Create data folder if it doesn't exist
        os.makedirs(self.data_folder, exist_ok=True)

        # Initialize findminfd api
        self.api = DataLoader()
        self.api.login_by_token(api_token=api_token)

    def fetch_stock_data(self, start_date, end_date=None):
        """
        Fetch stock data from Yahoo Finance API
        
        Parameters:
        - start_date (str): Start date in format 'YYYY-MM-DD'
        - end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to current date
        
        Returns:
        - pandas.DataFrame: DataFrame containing the stock data
        """
        try:
            # If end_date is not provided, use current date
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            print(f"Fetching {self.symbol} data from {start_date} to {end_date}")
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(self.symbol)
            data = stock.history(start=start_date, end=end_date, interval=self.interval)
            
            if data.empty:
                print(f"No data found for {self.symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to match the expected format
            data = data.rename(columns={
                'Date': 'open_time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            })
            
            # Add close_time column (end of day)
            data['close_time'] = data['open_time'] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Calculate quote asset volume (price * volume)
            data['quote_asset_volume'] = data['close'] * data['volume']
            
            return data
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def fetch_all_historical_data(self, start_date, end_date=None):
        """
        Fetch all historical stock data for the specified number of years back
        
        Parameters:
        - start_date (str): Start date in format 'YYYY-MM-DD'
        - end_date (str, optional): End date in format 'YYYY-MM-DD', defaults to current date
        
        Returns:
        - pandas.DataFrame: DataFrame containing all the stock data
        """
        # Use current date as end date
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching {self.symbol} daily data from {start_date} to {end_date}")
        
        # Fetch data from Yahoo Finance
        stock_data = self.fetch_stock_data(start_date, end_date)
        
        if stock_data.empty:
            print("No Stock data retrieved")
            return pd.DataFrame()
            
        # Sort by open_time
        stock_data = stock_data.sort_values('open_time')
        
        # Reset index
        stock_data = stock_data.reset_index(drop=True)
        
        return stock_data
    
    def get_file_size_mb(self, file_path):
        """
        Get the size of a file in megabytes
        
        Parameters:
        - file_path (str): Path to the file
        
        Returns:
        - float: Size of the file in megabytes
        """
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return 0
    
    def save_to_csv(self, data, base_filename=None):
        """
        Save the stock data to a CSV file in the data folder
        If the file size would exceed the maximum size, create a new file
        
        Parameters:
        - data (pandas.DataFrame): Stock data
        - base_filename (str, optional): Base name of the CSV file to save, defaults to symbol_interval
        
        Returns:
        - str: Path to the saved file
        """
        if data.empty:
            print("No data to save")
            return None
        
        # Default base filename
        if base_filename is None:
            base_filename = f"{self.symbol.replace('.', '_')}_{self.interval}_data"
            
        # Get current date for filename
        current_date = datetime.now().strftime("%Y%m%d")
        
        # Find existing files with the same base name
        existing_files = [f for f in os.listdir(self.data_folder) 
                         if f.startswith(f"{base_filename}_{current_date}") and f.endswith(".csv")]
        
        # Sort existing files to get the latest one
        existing_files.sort()
        
        if existing_files:
            latest_file = existing_files[-1]
            latest_file_path = os.path.join(self.data_folder, latest_file)
            
            # Check if adding to the latest file would exceed the size limit
            # Estimate the size of the new data
            buffer = io.StringIO()
            data.to_csv(buffer, index=False)
            buffer.seek(0)
            new_data_size_mb = sys.getsizeof(buffer.getvalue()) / (1024 * 1024)
            
            current_size_mb = self.get_file_size_mb(latest_file_path)
            
            if current_size_mb + new_data_size_mb <= self.max_file_size_mb:
                # check if there's any overlap with existing data
                existing_data = pd.read_csv(latest_file_path)
                
                if 'open_time' in existing_data.columns:
                    # Convert open_time to datetime for comparison
                    existing_data['open_time'] = pd.to_datetime(existing_data['open_time'])
                    
                    # Find the latest timestamp in the existing data
                    latest_timestamp = existing_data['open_time'].max()
                    
                    # Filter out data that's already in the existing file
                    if 'open_time' in data.columns:
                        data = data[data['open_time'] > latest_timestamp]
                
                # If there's new data to append
                if not data.empty:
                    # Append to the existing file
                    data.to_csv(latest_file_path, mode='a', header=False, index=False)
                    print(f"Data appended to {latest_file_path}")
                    return latest_file_path
                else:
                    print("No new data to append")
                    return latest_file_path

            file_index = int(latest_file.split('_')[-1].split('.')[0]) + 1
        else:
            # No existing files, start with index 1
            file_index = 1
        
        # Create a new file
        filename = f"{base_filename}_{current_date}_{file_index:03d}.csv"
        file_path = os.path.join(self.data_folder, filename)
        
        # Save the data
        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        
        return file_path


if __name__ == "__main__":
    # Create an instance of the NasdaqIndexFetcher
    fetcher = NasdaqIndexFetcher(data_folder="data", max_file_size_mb=50, api_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNC0xMSAxODo0MTo0OCIsInVzZXJfaWQiOiJjaHVjaHUwNzMxIiwiaXAiOiI0OS4yMTYuOTAuMTExIn0.wkmgH40VJ0vOJUluKXQk_tWKMeiNWXGFpWOSZsEZ9Nk')
    
    # Fetch historical data for the past 10 years
    start_date = '2021-01-01'
    end_date = '2024-01-01'
    stock_data = fetcher.fetch_all_historical_data(start_date, end_date)
    print(stock_data.columns)
    
    if not stock_data.empty:
        print(f"Retrieved {len(stock_data)} stock data records")
        
        # Display the first few records
        print("\nSample data:")
        print(stock_data.head())
        
        # Save to CSV
        csv_path = fetcher.save_to_csv(stock_data)
        
        # Print some statistics
        print("\nData Statistics:")
        print(f"Date Range: {stock_data['open_time'].min()} to {stock_data['open_time'].max()}")
        print(f"Price Range: {stock_data['low'].min()} to {stock_data['high'].max()}")
        print(f"Total Volume: {stock_data['volume'].sum():.2f} shares")
        print(f"Average Daily Volume: {stock_data['volume'].mean():.2f} shares")
    else:
        print("No stock data retrieved.")
