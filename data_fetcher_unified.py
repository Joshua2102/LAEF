"""
Data Fetcher Unified - Simple wrapper for Alpaca data fetching
Provides unified interface for market data fetching
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

class DataFetcher:
    """Unified data fetcher using Alpaca API"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if self.api_key and self.secret_key:
            self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        else:
            self.data_client = None
            print("Warning: Alpaca API credentials not found - data fetching will be limited")
    
    def fetch_stock_data(self, symbol: str, interval: str = '1h', period: str = '7d'):
        """
        Fetch stock data for given symbol and period
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            period: Period string ('1d', '7d', '1mo', '3mo', '6mo', '1y')
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.data_client:
            return pd.DataFrame()
        
        try:
            # Convert period to days
            days_map = {
                '1d': 1, '7d': 7, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365
            }
            days = days_map.get(period, 7)
            
            # Convert interval to Alpaca TimeFrame
            interval_map = {
                '1m': TimeFrame.Minute, '5m': TimeFrame(5, 'Min'),
                '15m': TimeFrame(15, 'Min'), '1h': TimeFrame.Hour,
                '1d': TimeFrame.Day
            }
            timeframe = interval_map.get(interval, TimeFrame.Hour)
            
            # Calculate start time
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            
            # Fetch data
            bars = self.data_client.get_stock_bars(request)
            
            if bars.df.empty:
                return pd.DataFrame()
            
            # Convert to standard format
            df = bars.df.copy()
            df = df.reset_index()
            
            # Ensure we have the right columns
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in expected_columns):
                print(f"Warning: Missing expected columns in data for {symbol}")
                return pd.DataFrame()
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

# Global fetcher instance
_fetcher_instance = None

def get_fetcher():
    """Get the global data fetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = DataFetcher()
    return _fetcher_instance

def fetch_stock_data(symbol: str, interval: str = '1h', period: str = '7d'):
    """Convenience function for fetching stock data"""
    fetcher = get_fetcher()
    return fetcher.fetch_stock_data(symbol, interval, period)

def get_latest_price(symbol: str):
    """
    Get the latest price for a symbol
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        
    Returns:
        float: Latest closing price or None if error
    """
    try:
        # Get the latest daily data point
        df = fetch_stock_data(symbol, interval='1d', period='1d')
        
        if df.empty:
            print(f"Warning: No data available for {symbol}")
            return None
            
        # Return the latest close price
        latest_price = df['close'].iloc[-1]
        return float(latest_price)
        
    except Exception as e:
        print(f"Error getting latest price for {symbol}: {e}")
        return None

def get_current_price(symbol: str):
    """Alias for get_latest_price for backward compatibility"""
    return get_latest_price(symbol)