"""
Experience Replay Buffer for Continuous Learning
Stores trading experiences and enables online learning from market outcomes.
"""

import numpy as np
import pandas as pd
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time

class TradingExperience:
    """Single trading experience record."""
    
    def __init__(self, timestamp: datetime, symbol: str, state: np.ndarray, 
                 q_value: float, action: str, price: float, confidence: float):
        self.timestamp = timestamp
        self.symbol = symbol
        self.state = state
        self.q_value = q_value
        self.action = action
        self.price = price
        self.confidence = confidence
        
        # Filled later when outcome is known
        self.actual_price_1h = None
        self.actual_price_1d = None
        self.actual_price_1w = None
        self.reward = None
        self.accuracy_score = None
        self.is_complete = False

class ExperienceReplayBuffer:
    """
    Experience replay buffer for continuous learning in trading.
    Stores trading experiences and calculates rewards based on actual market outcomes.
    """
    
    def __init__(self, max_size: int = 10000, min_experiences: int = 100):
        self.max_size = max_size
        self.min_experiences = min_experiences
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        # Tracking metrics
        self.total_experiences = 0
        self.completed_experiences = 0
        self.avg_accuracy_1h = 0.0
        self.avg_accuracy_1d = 0.0
        self.avg_accuracy_1w = 0.0
        
        logging.info(f"[EXPERIENCE] Buffer initialized with max_size={max_size}")
    
    def add_experience(self, timestamp: datetime, symbol: str, state: np.ndarray,
                      q_value: float, action: str, price: float, confidence: float):
        """Add a new trading experience to the buffer."""
        try:
            with self.lock:
                experience = TradingExperience(
                    timestamp=timestamp,
                    symbol=symbol,
                    state=state.copy(),
                    q_value=q_value,
                    action=action,
                    price=price,
                    confidence=confidence
                )
                
                self.buffer.append(experience)
                self.total_experiences += 1
                
                logging.debug(f"[EXPERIENCE] Added: {symbol} {action} @ ${price:.2f} (Q={q_value:.3f})")
                
        except Exception as e:
            logging.error(f"[EXPERIENCE] Failed to add experience: {e}")
    
    def update_outcomes(self, price_data: Dict[str, pd.DataFrame]):
        """
        Update experiences with actual market outcomes.
        
        Args:
            price_data: Dict of symbol -> DataFrame with timestamp, price columns
        """
        try:
            with self.lock:
                updated_count = 0
                
                for experience in self.buffer:
                    if experience.is_complete or experience.symbol not in price_data:
                        continue
                    
                    df = price_data[experience.symbol]
                    updated = self._update_single_experience(experience, df)
                    if updated:
                        updated_count += 1
                
                if updated_count > 0:
                    self._recalculate_metrics()
                    logging.info(f"[EXPERIENCE] Updated {updated_count} experiences with market outcomes")
                    
        except Exception as e:
            logging.error(f"[EXPERIENCE] Failed to update outcomes: {e}")
    
    def _update_single_experience(self, experience: TradingExperience, 
                                 price_df: pd.DataFrame) -> bool:
        """Update a single experience with actual market outcomes."""
        try:
            # Find prices at different time horizons
            base_time = experience.timestamp
            
            # 1 hour later
            time_1h = base_time + timedelta(hours=1)
            price_1h = self._get_closest_price(price_df, time_1h)
            
            # 1 day later
            time_1d = base_time + timedelta(days=1)
            price_1d = self._get_closest_price(price_df, time_1d)
            
            # 1 week later
            time_1w = base_time + timedelta(weeks=1)
            price_1w = self._get_closest_price(price_df, time_1w)
            
            # Only update if we have at least 1-day data
            if price_1d is None:
                return False
            
            experience.actual_price_1h = price_1h
            experience.actual_price_1d = price_1d
            experience.actual_price_1w = price_1w
            
            # Calculate accuracy and reward
            experience.accuracy_score = self._calculate_accuracy_score(experience)
            experience.reward = self._calculate_reward(experience)
            experience.is_complete = True
            
            self.completed_experiences += 1
            return True
            
        except Exception as e:
            logging.debug(f"[EXPERIENCE] Failed to update experience: {e}")
            return False
    
    def _get_closest_price(self, price_df: pd.DataFrame, target_time: datetime) -> Optional[float]:
        """Get the closest price to target time."""
        try:
            if price_df.empty:
                return None
            
            # Convert target time to match DataFrame index
            if hasattr(price_df.index, 'tz'):
                target_time = target_time.replace(tzinfo=price_df.index.tz)
            
            # Find closest timestamp
            time_diffs = abs(price_df.index - target_time)
            closest_idx = time_diffs.argmin()
            
            # Only use if within reasonable time window (e.g., 4 hours)
            if time_diffs.iloc[closest_idx] <= timedelta(hours=4):
                return float(price_df.iloc[closest_idx]['close'])
            
            return None
            
        except Exception as e:
            logging.debug(f"[EXPERIENCE] Failed to get closest price: {e}")
            return None
    
    def _calculate_accuracy_score(self, experience: TradingExperience) -> float:
        """Calculate prediction accuracy score based on Q-value vs actual outcomes."""
        try:
            base_price = experience.price
            q_value = experience.q_value
            
            # Calculate actual returns
            returns_1h = ((experience.actual_price_1h or base_price) - base_price) / base_price
            returns_1d = ((experience.actual_price_1d or base_price) - base_price) / base_price
            returns_1w = ((experience.actual_price_1w or base_price) - base_price) / base_price
            
            # Average return (weighted toward longer horizons)
            avg_return = (returns_1h * 0.2 + returns_1d * 0.5 + returns_1w * 0.3)
            
            # Q-value interpretation: >0.6 = bullish, <0.4 = bearish, 0.4-0.6 = neutral
            predicted_direction = 1 if q_value > 0.6 else (-1 if q_value < 0.4 else 0)
            actual_direction = 1 if avg_return > 0.02 else (-1 if avg_return < -0.02 else 0)
            
            # Calculate accuracy based on direction prediction
            if predicted_direction == actual_direction:
                accuracy = 1.0
            elif predicted_direction == 0 or actual_direction == 0:
                accuracy = 0.5  # Partial credit for neutral predictions
            else:
                accuracy = 0.0  # Wrong direction
            
            # Adjust based on confidence
            confidence_factor = experience.confidence
            return accuracy * confidence_factor
            
        except Exception as e:
            logging.debug(f"[EXPERIENCE] Failed to calculate accuracy: {e}")
            return 0.0
    
    def _calculate_reward(self, experience: TradingExperience) -> float:
        """Calculate reward based on prediction accuracy and market outcomes."""
        try:
            # Base reward from accuracy
            accuracy_reward = experience.accuracy_score * 1.0
            
            # Additional reward for correct high-confidence predictions
            if experience.accuracy_score > 0.8 and experience.confidence > 0.7:
                accuracy_reward *= 1.5
            
            # Penalty for wrong high-confidence predictions
            if experience.accuracy_score < 0.2 and experience.confidence > 0.7:
                accuracy_reward *= 0.5
            
            return accuracy_reward
            
        except Exception as e:
            logging.debug(f"[EXPERIENCE] Failed to calculate reward: {e}")
            return 0.0
    
    def _recalculate_metrics(self):
        """Recalculate buffer-wide metrics."""
        try:
            if self.completed_experiences == 0:
                return
            
            completed = [exp for exp in self.buffer if exp.is_complete]
            
            if completed:
                accuracies = [exp.accuracy_score for exp in completed]
                self.avg_accuracy_1d = np.mean(accuracies)
                
                logging.debug(f"[EXPERIENCE] Metrics updated: {len(completed)} complete, "
                             f"avg_accuracy={self.avg_accuracy_1d:.3f}")
                
        except Exception as e:
            logging.debug(f"[EXPERIENCE] Failed to recalculate metrics: {e}")
    
    def get_training_batch(self, batch_size: int = 32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get a batch of experiences for training."""
        try:
            with self.lock:
                completed = [exp for exp in self.buffer if exp.is_complete]
                
                if len(completed) < self.min_experiences:
                    return None
                
                # Sample batch
                indices = np.random.choice(len(completed), min(batch_size, len(completed)), replace=False)
                batch = [completed[i] for i in indices]
                
                # Prepare training data
                states = np.array([exp.state for exp in batch])
                targets = np.array([exp.q_value + exp.reward * 0.1 for exp in batch])  # Update rule
                
                return states, targets
                
        except Exception as e:
            logging.error(f"[EXPERIENCE] Failed to get training batch: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        with self.lock:
            return {
                'total_experiences': self.total_experiences,
                'completed_experiences': self.completed_experiences,
                'buffer_size': len(self.buffer),
                'completion_rate': self.completed_experiences / max(1, self.total_experiences),
                'avg_accuracy': self.avg_accuracy_1d,
                'ready_for_training': len([exp for exp in self.buffer if exp.is_complete]) >= self.min_experiences
            }
    
    def save_experiences(self, filepath: str):
        """Save experiences to file for analysis."""
        try:
            with self.lock:
                data = []
                for exp in self.buffer:
                    if exp.is_complete:
                        data.append({
                            'timestamp': exp.timestamp,
                            'symbol': exp.symbol,
                            'q_value': exp.q_value,
                            'action': exp.action,
                            'price': exp.price,
                            'confidence': exp.confidence,
                            'actual_price_1d': exp.actual_price_1d,
                            'accuracy_score': exp.accuracy_score,
                            'reward': exp.reward
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    df.to_csv(filepath, index=False)
                    logging.info(f"[EXPERIENCE] Saved {len(data)} experiences to {filepath}")
                    
        except Exception as e:
            logging.error(f"[EXPERIENCE] Failed to save experiences: {e}")