# === utils.py - Common utility functions for LAEF Trading System ===

import os
import logging
import pandas as pd
from typing import List, Optional
from datetime import datetime


def setup_logging(log_file: str = 'logs/laef.log') -> logging.Logger:
    """
    Set up centralized logging configuration.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_symbols_from_csv(ticker_file: str) -> List[str]:
    """
    Load ticker symbols from CSV file.
    
    Args:
        ticker_file: Path to CSV file containing ticker symbols
        
    Returns:
        List of ticker symbols
    """
    try:
        if not os.path.exists(ticker_file):
            logging.error(f"Ticker file not found: {ticker_file}")
            return []
        
        df = pd.read_csv(ticker_file)
        
        # Try different possible column names
        symbol_columns = ['ticker', 'Ticker', 'symbol', 'Symbol', 'SYMBOL']
        for col in symbol_columns:
            if col in df.columns:
                symbols = df[col].dropna().unique().tolist()
                logging.info(f"Loaded {len(symbols)} symbols from {ticker_file}")
                return symbols
        
        # If no recognized column, use first column
        if len(df.columns) > 0:
            symbols = df.iloc[:, 0].dropna().unique().tolist()
            logging.info(f"Loaded {len(symbols)} symbols from first column of {ticker_file}")
            return symbols
            
        logging.error(f"No valid symbol column found in {ticker_file}")
        return []
        
    except Exception as e:
        logging.error(f"Error loading symbols from {ticker_file}: {e}")
        return []


def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol is valid for trading.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation - alphanumeric, 1-5 characters
    if not symbol.isalnum() or len(symbol) > 5 or len(symbol) < 1:
        return False
    
    return True


def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Dollar amount
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a decimal as percentage.
    
    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def get_timestamp_suffix() -> str:
    """
    Get a timestamp suffix for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directories(*directories):
    """
    Ensure multiple directories exist.
    
    Args:
        *directories: Variable number of directory paths
    """
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)


def read_json_config(config_file: str) -> Optional[dict]:
    """
    Read a JSON configuration file.
    
    Args:
        config_file: Path to JSON config file
        
    Returns:
        Dictionary with config data or None if failed
    """
    try:
        import json
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to read config from {config_file}: {e}")
        return None


def write_json_config(config_data: dict, config_file: str) -> bool:
    """
    Write configuration data to JSON file.
    
    Args:
        config_data: Dictionary to save
        config_file: Path to JSON config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Failed to write config to {config_file}: {e}")
        return False