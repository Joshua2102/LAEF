# === agent_unified.py - Standardized Agent Implementation ===

import os
import numpy as np
import logging

# PERFORMANCE OPTIMIZATION: Configure TensorFlow before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
# Configure TensorFlow for better performance
tf.config.set_visible_devices([], 'GPU')  # Force CPU usage for consistency
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from config import STATE_SIZE, LEARNING_RATE, MODEL_PATH

class LAEFAgent:
    """
    Unified LAEF Agent with consistent state size and model architecture.
    """
    
    def __init__(self, state_size=None, model_path=None, pretrained=True):
        self.state_size = state_size or STATE_SIZE
        self.model_path = model_path or MODEL_PATH
        
        logging.info(f"[AGENT] Initializing with state_size={self.state_size}")
        
        # Build or load model
        self.model = self._build_model()
        
        if pretrained and os.path.exists(self.model_path):
            self.load_model()
        elif pretrained:
            logging.warning(f"[AGENT] Pretrained model not found at {self.model_path}")
    
    def _build_model(self):
        """
        Build neural network architecture optimized for trading signals.
        """
        try:
            model = Sequential([
                Input(shape=(self.state_size,)),
                Dense(128, activation='relu', name='hidden1'),
                Dropout(0.2),
                Dense(64, activation='relu', name='hidden2'), 
                Dropout(0.1),
                Dense(32, activation='relu', name='hidden3'),
                Dense(3, activation='linear', name='q_output')  # Q-values for [hold, buy, sell]
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss=MeanSquaredError(),
                metrics=['mae']
            )
            
            logging.info(f"[AGENT] Built model with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            logging.error(f"[AGENT] Failed to build model: {e}")
            raise
    
    def predict_q_values(self, state):
        """
        Predict Q-values for all actions [hold, buy, sell].
        
        Args:
            state: numpy array of shape (STATE_SIZE,) or (1, STATE_SIZE)
            
        Returns:
            numpy array: Q-values for [hold, buy, sell]
        """
        try:
            # Validate input
            if state is None:
                logging.warning("[AGENT] None state provided")
                return np.array([0.0, 0.0, 0.0])
            
            # Convert to numpy array if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
            # Ensure correct shape
            if state.ndim == 1:
                if len(state) != self.state_size:
                    logging.error(f"[AGENT] Wrong state size: {len(state)} vs {self.state_size}")
                    return np.array([0.0, 0.0, 0.0])
                state = state.reshape(1, -1)
            elif state.ndim == 2:
                if state.shape[1] != self.state_size:
                    logging.error(f"[AGENT] Wrong state shape: {state.shape}")
                    return np.array([0.0, 0.0, 0.0])
            else:
                logging.error(f"[AGENT] Invalid state dimensions: {state.ndim}")
                return np.array([0.0, 0.0, 0.0])
            
            # Validate values
            if np.any(np.isnan(state)):
                logging.warning("[AGENT] NaN values in state, replacing with 0")
                state = np.nan_to_num(state)
            
            if np.any(np.isinf(state)):
                logging.warning("[AGENT] Infinite values in state, clipping")
                state = np.clip(state, -10, 10)
            
            # Make prediction
            q_values = self.model.predict(state, verbose=0)[0]
            
            # Validate output
            if np.any(np.isnan(q_values)) or np.any(np.isinf(q_values)):
                logging.warning(f"[AGENT] Invalid Q-values predicted: {q_values}")
                return np.array([0.0, 0.0, 0.0])
            
            return q_values.astype(float)
            
        except Exception as e:
            logging.error(f"[AGENT] Prediction failed: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def predict_q_value(self, state):
        """
        Backward compatibility: Predict single Q-value (max of all actions).
        
        Args:
            state: numpy array of shape (STATE_SIZE,) or (1, STATE_SIZE)
            
        Returns:
            float: Maximum Q-value prediction
        """
        q_values = self.predict_q_values(state)
        return float(np.max(q_values))
    
    def predict_action(self, state):
        """
        Predict the best action for a given state.
        
        Args:
            state: numpy array of shape (STATE_SIZE,) or (1, STATE_SIZE)
            
        Returns:
            int: Action index (0=hold, 1=buy, 2=sell)
        """
        q_values = self.predict_q_values(state)
        return int(np.argmax(q_values))
    
    def predict_q_values_batch(self, states):
        """
        PERFORMANCE OPTIMIZATION: Batch prediction for multiple states.
        Much faster than calling predict_q_values multiple times.
        
        Args:
            states: List or array of states
            
        Returns:
            Array of Q-values with shape (n_samples, 3)
        """
        try:
            # Convert to numpy array
            states = np.array(states)
            
            # Ensure correct shape
            if states.ndim == 2 and states.shape[1] == self.state_size:
                # Predict all at once - MUCH faster than one-by-one
                predictions = self.model.predict(states, batch_size=32, verbose=0)
                return predictions
            else:
                logging.error(f"[AGENT] Invalid batch shape: {states.shape}")
                return np.zeros((len(states), 3))
                
        except Exception as e:
            logging.error(f"[AGENT] Batch prediction failed: {e}")
            return np.zeros((len(states), 3))
    
    def train(self, states, targets, epochs=None, batch_size=None, validation_split=0.1):
        """
        Train the model on provided data.
        """
        try:
            from config import EPOCHS, BATCH_SIZE
            epochs = epochs or EPOCHS
            batch_size = batch_size or BATCH_SIZE
            
            logging.info(f"[AGENT] Training on {len(states)} samples for {epochs} epochs")
            
            # Validate inputs
            states = np.array(states)
            targets = np.array(targets)
            
            if states.shape[1] != self.state_size:
                raise ValueError(f"State size mismatch: {states.shape[1]} vs {self.state_size}")
            
            # Train model
            history = self.model.fit(
                states, targets,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            logging.info(f"[AGENT] Training completed. Final loss: {history.history['loss'][-1]:.4f}")
            return history
            
        except Exception as e:
            logging.error(f"[AGENT] Training failed: {e}")
            raise
    
    def save_model(self, path=None):
        """
        Save model to disk.
        """
        try:
            save_path = path or self.model_path
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            self.model.save(save_path)
            logging.info(f"[AGENT] Model saved to: {save_path}")
            
        except Exception as e:
            logging.error(f"[AGENT] Failed to save model: {e}")
            raise
    
    def load_model(self, path=None):
        """
        Load model from disk, with fallback to building new model if load fails.
        """
        try:
            load_path = path or self.model_path
            
            if not os.path.exists(load_path):
                logging.warning(f"[AGENT] Model not found: {load_path}, using fresh model")
                return
            
            # Try to load with error handling for compatibility issues
            try:
                self.model = load_model(load_path)
                logging.info(f"[AGENT] Model loaded from: {load_path}")
                
                # Verify model architecture
                if self.model.input_shape[1] != self.state_size:
                    logging.warning(f"[AGENT] Loaded model state size ({self.model.input_shape[1]}) doesn't match expected ({self.state_size})")
                    # Rebuild model with correct architecture
                    logging.info("[AGENT] Rebuilding model with correct state size")
                    self.model = self._build_model()
                    
            except Exception as load_error:
                logging.warning(f"[AGENT] Model loading failed ({load_error}), rebuilding fresh model")
                # If loading fails, rebuild the model
                self.model = self._build_model()
            
        except Exception as e:
            logging.error(f"[AGENT] Failed to load model: {e}")
            # Don't raise - just use the fresh model that was built
    
    def get_model_info(self):
        """
        Get information about the current model.
        """
        return {
            'state_size': self.state_size,
            'model_path': self.model_path,
            'total_params': self.model.count_params(),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape
        }

class AgentWrapper:
    """
    Wrapper class for backward compatibility with existing code.
    """
    
    def __init__(self, state_size=None, model_path=None, pretrained=True):
        self.agent = LAEFAgent(state_size, model_path, pretrained)
    
    def predict_q_value(self, state):
        return self.agent.predict_q_value(state)