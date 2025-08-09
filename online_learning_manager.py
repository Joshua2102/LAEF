"""
Online Learning Manager for Continuous Model Training
Manages the continuous learning process for the LAEF trading system.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import pandas as pd

from core.experience_buffer import ExperienceReplayBuffer
from core.agent_unified import LAEFAgent
from data.data_fetcher_unified import get_fetcher
from config import ENABLE_AUTO_RETRAINING, BATCH_SIZE, LEARNING_RATE

class OnlineLearningManager:
    """
    Manages continuous learning for the trading system.
    Collects experiences, updates outcomes, and retrains the model.
    """
    
    def __init__(self, agent: LAEFAgent, symbols: list, learning_interval: int = 3600):
        """
        Initialize the online learning manager.
        
        Args:
            agent: The LAEF agent to train
            symbols: List of symbols to monitor
            learning_interval: Seconds between learning updates (default: 1 hour)
        """
        self.agent = agent
        self.symbols = symbols
        self.learning_interval = learning_interval
        
        # Experience buffer
        self.experience_buffer = ExperienceReplayBuffer(max_size=10000, min_experiences=50)
        
        # Data fetcher for getting market outcomes
        self.data_fetcher = get_fetcher()
        
        # Learning control
        self.is_learning = False
        self.learning_thread = None
        self.last_learning_time = datetime.now()
        self.last_data_update = datetime.now()
        
        # Performance tracking
        self.learning_stats = {
            'total_training_sessions': 0,
            'total_experiences_processed': 0,
            'last_training_loss': None,
            'avg_prediction_accuracy': 0.0,
            'model_updates': 0
        }
        
        logging.info(f"[LEARNING] Online Learning Manager initialized")
        logging.info(f"[LEARNING] Monitoring {len(symbols)} symbols: {', '.join(symbols[:5])}...")
        logging.info(f"[LEARNING] Learning interval: {learning_interval}s ({learning_interval/3600:.1f}h)")
    
    def start_learning(self):
        """Start the continuous learning process."""
        if self.is_learning:
            logging.warning("[LEARNING] Learning already active")
            return
        
        if not ENABLE_AUTO_RETRAINING:
            logging.info("[LEARNING] Auto-retraining disabled in config")
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        logging.info("[LEARNING] Continuous learning started")
    
    def stop_learning(self):
        """Stop the continuous learning process."""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=30)
        
        logging.info("[LEARNING] Continuous learning stopped")
    
    def add_trading_experience(self, symbol: str, state: np.ndarray, q_value: float,
                              action: str, price: float, confidence: float):
        """
        Add a trading decision to the experience buffer.
        
        Args:
            symbol: Stock symbol
            state: Market state vector
            q_value: Predicted Q-value
            action: Trading action taken
            price: Current price
            confidence: Decision confidence
        """
        try:
            timestamp = datetime.now()
            self.experience_buffer.add_experience(
                timestamp=timestamp,
                symbol=symbol,
                state=state,
                q_value=q_value,
                action=action,
                price=price,
                confidence=confidence
            )
            
            logging.debug(f"[LEARNING] Experience added: {symbol} {action} @ ${price:.2f}")
            
        except Exception as e:
            logging.error(f"[LEARNING] Failed to add experience: {e}")
    
    def _learning_loop(self):
        """Main learning loop running in background thread."""
        logging.info("[LEARNING] Learning loop started")
        
        while self.is_learning:
            try:
                current_time = datetime.now()
                
                # Check if it's time for a learning update
                if (current_time - self.last_learning_time).seconds >= self.learning_interval:
                    self._perform_learning_update()
                    self.last_learning_time = current_time
                
                # Update market outcomes more frequently (every 15 minutes)
                if (current_time - self.last_data_update).seconds >= 900:  # 15 minutes
                    self._update_market_outcomes()
                    self.last_data_update = current_time
                
                # Sleep for a minute before next check
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"[LEARNING] Error in learning loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _perform_learning_update(self):
        """Perform a learning update if enough experiences are available."""
        try:
            # Get training batch
            batch_data = self.experience_buffer.get_training_batch(batch_size=BATCH_SIZE)
            
            if batch_data is None:
                stats = self.experience_buffer.get_stats()
                logging.debug(f"[LEARNING] Not enough experiences for training "
                             f"({stats['completed_experiences']}/{stats['total_experiences']} complete)")
                return
            
            states, targets = batch_data
            
            # Perform training
            logging.info(f"[LEARNING] Starting training session with {len(states)} experiences")
            
            history = self.agent.train(
                states=states,
                targets=targets,
                epochs=1,  # Single epoch for online learning
                batch_size=min(BATCH_SIZE, len(states)),
                validation_split=0.0  # No validation split for online learning
            )
            
            # Update statistics
            self.learning_stats['total_training_sessions'] += 1
            self.learning_stats['total_experiences_processed'] += len(states)
            self.learning_stats['last_training_loss'] = history.history['loss'][-1]
            self.learning_stats['model_updates'] += 1
            
            # Update prediction accuracy
            buffer_stats = self.experience_buffer.get_stats()
            self.learning_stats['avg_prediction_accuracy'] = buffer_stats['avg_accuracy']
            
            # Save updated model
            self.agent.save_model()
            
            logging.info(f"[LEARNING] Training completed - Loss: {self.learning_stats['last_training_loss']:.4f}, "
                        f"Accuracy: {self.learning_stats['avg_prediction_accuracy']:.3f}")
            
        except Exception as e:
            logging.error(f"[LEARNING] Training failed: {e}")
    
    def _update_market_outcomes(self):
        """Update experiences with latest market data."""
        try:
            logging.debug("[LEARNING] Updating market outcomes...")
            
            # Fetch recent price data for all symbols
            price_data = {}
            current_time = datetime.now()
            
            for symbol in self.symbols:
                try:
                    # Get last 7 days of data to capture outcomes
                    df = self.data_fetcher.fetch_stock_data(
                        symbol=symbol,
                        interval='1h',
                        period='7d'
                    )
                    
                    if df is not None and not df.empty:
                        price_data[symbol] = df
                        
                except Exception as e:
                    logging.debug(f"[LEARNING] Failed to fetch data for {symbol}: {e}")
            
            if price_data:
                self.experience_buffer.update_outcomes(price_data)
                logging.debug(f"[LEARNING] Updated outcomes for {len(price_data)} symbols")
            
        except Exception as e:
            logging.error(f"[LEARNING] Failed to update market outcomes: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics."""
        buffer_stats = self.experience_buffer.get_stats()
        
        return {
            'is_learning': self.is_learning,
            'learning_interval_hours': self.learning_interval / 3600,
            'symbols_monitored': len(self.symbols),
            'buffer_stats': buffer_stats,
            'training_stats': self.learning_stats,
            'last_learning_time': self.last_learning_time.isoformat(),
            'last_data_update': self.last_data_update.isoformat()
        }
    
    def force_learning_update(self):
        """Force an immediate learning update (for testing/debugging)."""
        logging.info("[LEARNING] Forcing immediate learning update...")
        self._update_market_outcomes()
        self._perform_learning_update()
    
    def save_learning_data(self, base_path: str = "logs"):
        """Save learning data for analysis."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save experiences
            experience_file = f"{base_path}/learning_experiences_{timestamp}.csv"
            self.experience_buffer.save_experiences(experience_file)
            
            # Save learning stats
            stats_file = f"{base_path}/learning_stats_{timestamp}.json"
            import json
            with open(stats_file, 'w') as f:
                json.dump(self.get_learning_stats(), f, indent=2, default=str)
            
            logging.info(f"[LEARNING] Learning data saved to {base_path}/learning_*_{timestamp}.*")
            
        except Exception as e:
            logging.error(f"[LEARNING] Failed to save learning data: {e}")
    
    def get_prediction_confidence_trend(self, days: int = 7) -> Dict:
        """Get prediction confidence trend over recent days."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with self.experience_buffer.lock:
                recent_experiences = [
                    exp for exp in self.experience_buffer.buffer 
                    if exp.is_complete and exp.timestamp >= cutoff_time
                ]
            
            if not recent_experiences:
                return {'trend': 'no_data', 'accuracy': 0.0, 'count': 0}
            
            # Calculate daily accuracy
            daily_accuracy = {}
            for exp in recent_experiences:
                day_key = exp.timestamp.date()
                if day_key not in daily_accuracy:
                    daily_accuracy[day_key] = []
                daily_accuracy[day_key].append(exp.accuracy_score)
            
            # Average by day
            daily_averages = {
                day: np.mean(scores) for day, scores in daily_accuracy.items()
            }
            
            if len(daily_averages) < 2:
                return {'trend': 'insufficient_data', 'accuracy': np.mean([exp.accuracy_score for exp in recent_experiences]), 'count': len(recent_experiences)}
            
            # Calculate trend
            dates = sorted(daily_averages.keys())
            accuracies = [daily_averages[date] for date in dates]
            
            # Simple linear trend
            x = np.arange(len(accuracies))
            z = np.polyfit(x, accuracies, 1)
            trend_slope = z[0]
            
            trend = 'improving' if trend_slope > 0.01 else ('declining' if trend_slope < -0.01 else 'stable')
            
            return {
                'trend': trend,
                'slope': trend_slope,
                'current_accuracy': accuracies[-1],
                'avg_accuracy': np.mean(accuracies),
                'count': len(recent_experiences),
                'days_analyzed': len(daily_averages)
            }
            
        except Exception as e:
            logging.error(f"[LEARNING] Failed to calculate confidence trend: {e}")
            return {'trend': 'error', 'accuracy': 0.0, 'count': 0}

# Global learning manager instance
_learning_manager = None

def get_learning_manager() -> Optional[OnlineLearningManager]:
    """Get the global learning manager instance."""
    return _learning_manager

def initialize_learning_manager(agent: LAEFAgent, symbols: list) -> OnlineLearningManager:
    """Initialize the global learning manager."""
    global _learning_manager
    _learning_manager = OnlineLearningManager(agent, symbols)
    return _learning_manager