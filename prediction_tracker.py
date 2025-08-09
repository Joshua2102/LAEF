"""
Standalone Prediction Tracker for Live Market Learning
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List


class PredictionTracker:
    """
    Tracks predictions and their outcomes in a SQLite database
    """
    
    def __init__(self, db_path: str = "logs/training/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize prediction tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                prediction_type TEXT,
                timeframe TEXT,
                current_price REAL,
                predicted_price REAL,
                predicted_direction TEXT,
                confidence REAL,
                q_value REAL,
                ml_score REAL,
                technical_indicators TEXT,
                market_conditions TEXT,
                outcome_price REAL,
                outcome_timestamp DATETIME,
                actual_return REAL,
                prediction_accuracy TEXT,
                learning_applied BOOLEAN DEFAULT 0
            )
        ''')
        
        # Create learning history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                prediction_id INTEGER,
                weight_updates TEXT,
                performance_before REAL,
                performance_after REAL,
                learning_rate REAL,
                notes TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON predictions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_accuracy ON predictions(prediction_accuracy)')
        
        conn.commit()
        conn.close()
    
    def add_prediction(self, symbol: str, prediction_type: str, timeframe: str,
                      current_price: float, predicted_price: float, 
                      confidence: float, q_value: float, ml_score: float,
                      indicators: Dict, market_conditions: Dict) -> int:
        """Add a new prediction to track"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        predicted_direction = 'up' if predicted_price > current_price else 'down' if predicted_price < current_price else 'neutral'
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, symbol, prediction_type, timeframe, current_price, 
             predicted_price, predicted_direction, confidence, q_value, ml_score,
             technical_indicators, market_conditions, prediction_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(), symbol, prediction_type, timeframe, current_price,
            predicted_price, predicted_direction, confidence, q_value, ml_score,
            json.dumps(indicators), json.dumps(market_conditions), 'pending'
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_outcome(self, prediction_id: int, outcome_price: float):
        """Update prediction with actual outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
        row = cursor.fetchone()
        
        if row:
            current_price = row[5]
            predicted_price = row[6]
            predicted_direction = row[7]
            
            actual_return = (outcome_price - current_price) / current_price
            actual_direction = 'up' if outcome_price > current_price else 'down' if outcome_price < current_price else 'neutral'
            
            price_error = abs(predicted_price - outcome_price) / current_price
            direction_correct = predicted_direction == actual_direction
            
            if direction_correct and price_error < 0.02:
                accuracy = 'correct'
            elif direction_correct:
                accuracy = 'partially_correct'
            else:
                accuracy = 'incorrect'
            
            cursor.execute('''
                UPDATE predictions 
                SET outcome_price = ?, outcome_timestamp = ?, 
                    actual_return = ?, prediction_accuracy = ?
                WHERE id = ?
            ''', (outcome_price, datetime.now(), actual_return, accuracy, prediction_id))
            
            conn.commit()
        
        conn.close()
    
    def get_pending_predictions(self, timeframe: str = None) -> List[Dict]:
        """Get predictions that need outcome updates"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM predictions WHERE prediction_accuracy = "pending"'
        
        if timeframe:
            query += ' AND timeframe = ?'
            cursor.execute(query, (timeframe,))
        else:
            cursor.execute(query)
        
        predictions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return predictions
    
    def get_performance_stats(self, symbol: str = None, days: int = 30) -> Dict:
        """Get prediction performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days)
        
        base_query = '''
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN prediction_accuracy = "correct" THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN prediction_accuracy = "partially_correct" THEN 1 ELSE 0 END) as partial,
                SUM(CASE WHEN prediction_accuracy = "incorrect" THEN 1 ELSE 0 END) as incorrect,
                AVG(confidence) as avg_confidence,
                AVG(ABS(predicted_price - outcome_price) / current_price) as avg_price_error
            FROM predictions
            WHERE prediction_accuracy != "pending"
            AND timestamp > ?
        '''
        
        if symbol:
            cursor.execute(base_query + ' AND symbol = ?', (since_date, symbol))
        else:
            cursor.execute(base_query, (since_date,))
        
        stats = cursor.fetchone()
        conn.close()
        
        if stats and stats[0] > 0:
            return {
                'total_predictions': stats[0],
                'accuracy_rate': (stats[1] + stats[2] * 0.5) / stats[0],
                'correct': stats[1],
                'partial': stats[2],
                'incorrect': stats[3],
                'avg_confidence': stats[4] or 0,
                'avg_price_error': stats[5] or 0
            }
        else:
            return {
                'total_predictions': 0,
                'accuracy_rate': 0,
                'correct': 0,
                'partial': 0,
                'incorrect': 0,
                'avg_confidence': 0,
                'avg_price_error': 0
            }
    
    def get_learning_candidates(self, min_predictions: int = 20) -> List[Dict]:
        """Get completed predictions ready for learning"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE prediction_accuracy != "pending" 
            AND learning_applied = 0
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (min_predictions * 2,))  # Get more than needed
        
        candidates = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return candidates[:min_predictions] if len(candidates) >= min_predictions else candidates
    
    def mark_learning_applied(self, prediction_ids: List[int]):
        """Mark predictions as having been used for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pred_id in prediction_ids:
            cursor.execute('''
                UPDATE predictions 
                SET learning_applied = 1 
                WHERE id = ?
            ''', (pred_id,))
        
        conn.commit()
        conn.close()