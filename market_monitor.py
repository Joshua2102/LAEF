"""
Continuous Market Data Monitor
Continuously monitors stock prices and feeds data to the learning system.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

from data.data_fetcher_unified import get_fetcher
from core.state_utils_unified import create_state_vector, extract_raw_indicators
from core.online_learning_manager import get_learning_manager
from config import DEFAULT_SYMBOL_UNIVERSE

class MarketDataMonitor:
    """
    Continuously monitors market data and feeds it to the learning system.
    Operates independently of trading activities.
    """
    
    def __init__(self, symbols: List[str] = None, monitoring_interval: int = 300):
        """
        Initialize market data monitor.
        
        Args:
            symbols: List of symbols to monitor (defaults to config symbols)
            monitoring_interval: Seconds between monitoring cycles (default: 5 minutes)
        """
        self.symbols = symbols or DEFAULT_SYMBOL_UNIVERSE[:20]  # Limit to top 20 for performance
        self.monitoring_interval = monitoring_interval
        self.data_fetcher = get_fetcher()
        
        # Monitoring control
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_monitoring_time = datetime.now()
        
        # Data storage
        self.latest_data = {}  # symbol -> latest DataFrame
        self.price_history = {}  # symbol -> price history for tracking
        self.monitoring_stats = {
            'total_monitoring_cycles': 0,
            'successful_symbol_updates': 0,
            'failed_symbol_updates': 0,
            'last_monitoring_time': None,
            'symbols_with_data': set(),
            'data_freshness': {}  # symbol -> last update time
        }
        
        logging.info(f"[MONITOR] Market Data Monitor initialized")
        logging.info(f"[MONITOR] Monitoring {len(self.symbols)} symbols: {', '.join(self.symbols[:5])}...")
        logging.info(f"[MONITOR] Monitoring interval: {monitoring_interval}s ({monitoring_interval/60:.1f}m)")
    
    def start_monitoring(self):
        """Start continuous market data monitoring."""
        if self.is_monitoring:
            logging.warning("[MONITOR] Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info("[MONITOR] Continuous market monitoring started")
    
    def stop_monitoring(self):
        """Stop market data monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        logging.info("[MONITOR] Market monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        logging.info("[MONITOR] Monitoring loop started")
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # Check if it's time for monitoring update
                if (current_time - self.last_monitoring_time).seconds >= self.monitoring_interval:
                    self._perform_monitoring_cycle()
                    self.last_monitoring_time = current_time
                
                # Sleep for 30 seconds before next check
                time.sleep(30)
                
            except Exception as e:
                logging.error(f"[MONITOR] Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _perform_monitoring_cycle(self):
        """Perform a complete monitoring cycle for all symbols."""
        try:
            logging.debug("[MONITOR] Starting monitoring cycle")
            cycle_start = datetime.now()
            
            successful_updates = 0
            failed_updates = 0
            
            for symbol in self.symbols:
                try:
                    success = self._update_symbol_data(symbol)
                    if success:
                        successful_updates += 1
                        self.monitoring_stats['symbols_with_data'].add(symbol)
                        self.monitoring_stats['data_freshness'][symbol] = datetime.now()
                    else:
                        failed_updates += 1
                        
                    # Small delay to avoid overwhelming APIs
                    time.sleep(0.5)
                    
                except Exception as e:
                    logging.debug(f"[MONITOR] Failed to update {symbol}: {e}")
                    failed_updates += 1
            
            # Update statistics
            self.monitoring_stats['total_monitoring_cycles'] += 1
            self.monitoring_stats['successful_symbol_updates'] += successful_updates
            self.monitoring_stats['failed_symbol_updates'] += failed_updates
            self.monitoring_stats['last_monitoring_time'] = datetime.now()
            
            cycle_duration = (datetime.now() - cycle_start).seconds
            
            logging.info(f"[MONITOR] Cycle complete: {successful_updates}/{len(self.symbols)} symbols updated "
                        f"in {cycle_duration}s ({failed_updates} failed)")
            
            # Feed data to learning system if available
            self._feed_learning_system()
            
        except Exception as e:
            logging.error(f"[MONITOR] Monitoring cycle failed: {e}")
    
    def _update_symbol_data(self, symbol: str) -> bool:
        """Update data for a single symbol."""
        try:
            # Fetch recent data (daily intervals for stable predictions)
            df = self.data_fetcher.fetch_stock_data(
                symbol=symbol,
                interval='1d',
                period='1y'
            )
            
            if df is None or df.empty:
                return False
            
            # Store latest data
            self.latest_data[symbol] = df
            
            # Update price history for tracking
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            latest_price = df['close'].iloc[-1]
            latest_time = df.index[-1]
            
            self.price_history[symbol].append({
                'timestamp': latest_time,
                'price': latest_price,
                'volume': df['volume'].iloc[-1]
            })
            
            # Keep only last 24 hours of price history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.price_history[symbol] = [
                entry for entry in self.price_history[symbol]
                if entry['timestamp'] >= cutoff_time
            ]
            
            return True
            
        except Exception as e:
            logging.debug(f"[MONITOR] Failed to update {symbol}: {e}")
            return False
    
    def _feed_learning_system(self):
        """Feed monitoring data to the learning system for experience collection."""
        try:
            learning_manager = get_learning_manager()
            if not learning_manager:
                return
            
            # Generate synthetic trading experiences from monitoring data
            for symbol in self.symbols:
                if symbol not in self.latest_data:
                    continue
                
                try:
                    df = self.latest_data[symbol]
                    if len(df) < 50:  # Need enough data for state vector
                        continue
                    
                    # Create state vector from latest data
                    window = df.tail(50)
                    state = create_state_vector(window)
                    
                    if state is None:
                        continue
                    
                    # Get current price and indicators
                    current_price = df['close'].iloc[-1]
                    indicators = extract_raw_indicators(window)
                    
                    # Generate prediction from current model (without taking action)
                    q_value = learning_manager.agent.predict_q_value(state)
                    
                    # Determine what action would be taken (for tracking purposes)
                    action = 'hold'  # Default for monitoring
                    confidence = 0.5  # Default confidence for monitoring
                    
                    if q_value > 0.65:
                        action = 'buy'
                        confidence = min(0.9, (q_value - 0.5) * 2)
                    elif q_value < 0.35:
                        action = 'sell'
                        confidence = min(0.9, (0.5 - q_value) * 2)
                    
                    # Add monitoring experience (this will be used to track prediction accuracy)
                    learning_manager.add_trading_experience(
                        symbol=symbol,
                        state=state,
                        q_value=q_value,
                        action=f"monitor_{action}",  # Prefix to distinguish from actual trades
                        price=current_price,
                        confidence=confidence
                    )
                    
                    logging.debug(f"[MONITOR] Added monitoring experience: {symbol} "
                                 f"monitor_{action} @ ${current_price:.2f} (Q={q_value:.3f})")
                    
                except Exception as e:
                    logging.debug(f"[MONITOR] Failed to feed learning system for {symbol}: {e}")
                    
        except Exception as e:
            logging.debug(f"[MONITOR] Failed to feed learning system: {e}")
    
    def get_monitoring_stats(self) -> Dict:
        """Get comprehensive monitoring statistics."""
        return {
            'is_monitoring': self.is_monitoring,
            'monitoring_interval_minutes': self.monitoring_interval / 60,
            'symbols_monitored': len(self.symbols),
            'symbols_with_current_data': len(self.monitoring_stats['symbols_with_data']),
            'total_cycles': self.monitoring_stats['total_monitoring_cycles'],
            'successful_updates': self.monitoring_stats['successful_symbol_updates'],
            'failed_updates': self.monitoring_stats['failed_symbol_updates'],
            'success_rate': (
                self.monitoring_stats['successful_symbol_updates'] / 
                max(1, self.monitoring_stats['successful_symbol_updates'] + self.monitoring_stats['failed_symbol_updates'])
            ) * 100,
            'last_monitoring_time': self.monitoring_stats['last_monitoring_time'].isoformat() 
                                   if self.monitoring_stats['last_monitoring_time'] else None,
            'data_freshness': {
                symbol: timestamp.isoformat() 
                for symbol, timestamp in self.monitoring_stats['data_freshness'].items()
            }
        }
    
    def get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get the latest data for a specific symbol."""
        return self.latest_data.get(symbol)
    
    def get_price_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get price history for a symbol."""
        if symbol not in self.price_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            entry for entry in self.price_history[symbol]
            if entry['timestamp'] >= cutoff_time
        ]
    
    def get_market_summary(self) -> Dict:
        """Get a summary of current market conditions."""
        try:
            summary = {
                'total_symbols': len(self.symbols),
                'active_symbols': 0,
                'average_price_change_1h': 0.0,
                'high_volatility_symbols': [],
                'last_update': datetime.now().isoformat()
            }
            
            price_changes = []
            
            for symbol in self.symbols:
                if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                    continue
                
                summary['active_symbols'] += 1
                
                # Calculate 1-hour price change
                recent_prices = sorted(self.price_history[symbol], key=lambda x: x['timestamp'])
                if len(recent_prices) >= 2:
                    hour_ago = datetime.now() - timedelta(hours=1)
                    
                    # Find prices closest to 1 hour ago and now
                    old_price = recent_prices[0]['price']
                    current_price = recent_prices[-1]['price']
                    
                    price_change = ((current_price - old_price) / old_price) * 100
                    price_changes.append(price_change)
                    
                    # Flag high volatility (>5% change in 1 hour)
                    if abs(price_change) > 5.0:
                        summary['high_volatility_symbols'].append({
                            'symbol': symbol,
                            'change_pct': price_change,
                            'current_price': current_price
                        })
            
            if price_changes:
                summary['average_price_change_1h'] = np.mean(price_changes)
            
            return summary
            
        except Exception as e:
            logging.error(f"[MONITOR] Failed to generate market summary: {e}")
            return {'error': str(e)}

# Global monitor instance
_market_monitor = None

def get_market_monitor() -> Optional[MarketDataMonitor]:
    """Get the global market monitor instance."""
    return _market_monitor

def initialize_market_monitor(symbols: List[str] = None) -> MarketDataMonitor:
    """Initialize the global market monitor."""
    global _market_monitor
    _market_monitor = MarketDataMonitor(symbols)
    return _market_monitor