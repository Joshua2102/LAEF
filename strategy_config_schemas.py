# === strategy_config_schemas.py - Centralized Strategy Configuration Schemas ===

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class BaseStrategyConfig:
    """Base configuration for all strategies"""
    
    # Common parameters
    max_position_size: float = 0.1  # Maximum position as % of portfolio
    max_concurrent_positions: int = 5  # Maximum number of positions
    cooldown_seconds: int = 60  # Cooldown between trades
    stop_loss_pct: float = 0.97  # Stop loss percentage (3% loss)
    trailing_stop_pct: float = 0.02  # Trailing stop percentage
    
    # Risk management
    max_risk_per_trade: float = 0.02  # Maximum risk per trade (2%)
    max_daily_loss: float = 0.05  # Maximum daily loss (5%)
    max_drawdown: float = 0.10  # Maximum drawdown (10%)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@dataclass
class AIStrategyConfig(BaseStrategyConfig):
    """Configuration for AI-based strategies"""
    
    # Q-Learning parameters
    q_buy_threshold: float = 0.30
    q_sell_threshold: float = 0.20
    q_learning_rate: float = 0.1
    q_discount_factor: float = 0.95
    
    # ML parameters
    ml_confidence_threshold: float = 0.30
    ml_profit_peak_threshold: float = 0.30
    ml_model_update_frequency: int = 100  # Update every N trades
    
    # AI-specific risk
    ai_confidence_weight: float = 0.7  # Weight for AI signals

@dataclass
class MomentumStrategyConfig(BaseStrategyConfig):
    """Configuration for momentum-based strategies"""
    
    # Momentum thresholds
    momentum_buy_threshold: float = 2.0  # 2% momentum
    momentum_sell_threshold: float = -1.0  # -1% momentum
    momentum_acceleration_min: float = 0.5
    
    # Scalping parameters
    micro_profit_target: float = 0.15  # 0.15% profit
    micro_stop_loss: float = 0.08  # 0.08% stop loss
    max_hold_minutes: int = 30  # Maximum hold time
    
    # Volume requirements
    volume_confirmation_min: float = 1.5  # 1.5x average volume
    volume_spike_threshold: float = 2.0  # 2x volume spike
    
    # Technical indicators
    roc_period: int = 5
    velocity_period: int = 3
    volume_sma_period: int = 10

@dataclass
class HybridStrategyConfig(BaseStrategyConfig):
    """Configuration for hybrid day/swing trading strategies"""
    
    # Day trading parameters
    day_trade_profit_target: float = 0.015  # 1.5% target
    day_trade_stop_loss: float = 0.007  # 0.7% stop
    max_day_trades: int = 3
    
    # Swing trading parameters
    swing_trade_profit_target: float = 0.04  # 4% target
    swing_trade_stop_loss: float = 0.02  # 2% stop
    max_swing_hold_days: int = 7
    
    # Entry conditions
    momentum_threshold: float = 0.02  # 2% momentum
    rsi_oversold: int = 20
    rsi_overbought: int = 80
    
    # Risk parameters
    max_risk_per_trade: float = 0.08  # 8% risk (aggressive)
    max_position_size: float = 0.25  # 25% position

@dataclass
class DualModelStrategyConfig(AIStrategyConfig, MomentumStrategyConfig):
    """Configuration for dual model (AI + Momentum) strategies"""
    
    # Combines AI and Momentum parameters
    # Strategy blending
    ai_weight: float = 0.6  # Weight for AI signals
    momentum_weight: float = 0.4  # Weight for momentum signals
    
    # Enhanced thresholds
    combined_signal_threshold: float = 0.6  # Minimum combined signal

class StrategyConfigManager:
    """Manages strategy configurations with validation and persistence"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.config_schemas = {
            'base': BaseStrategyConfig,
            'ai': AIStrategyConfig,
            'momentum': MomentumStrategyConfig,
            'hybrid': HybridStrategyConfig,
            'dual_model': DualModelStrategyConfig
        }
        
        # Default configurations
        self.default_configs = {
            'ai_momentum_scalping': DualModelStrategyConfig(
                q_buy_threshold=0.25,
                q_sell_threshold=0.15,
                ml_confidence_threshold=0.25,
                momentum_buy_threshold=1.0,
                micro_profit_target=0.12,
                micro_stop_loss=0.06,
                max_hold_minutes=45,
                max_position_size=0.025  # 2.5% for scalping
            ),
            'momentum_scalping': MomentumStrategyConfig(
                momentum_buy_threshold=2.0,
                momentum_sell_threshold=-1.0,
                micro_profit_target=0.15,
                micro_stop_loss=0.08,
                max_hold_minutes=30,
                max_position_size=0.02  # 2% for pure scalping
            ),
            'hybrid_trading': HybridStrategyConfig(
                day_trade_profit_target=0.015,
                swing_trade_profit_target=0.04,
                max_risk_per_trade=0.08,
                max_position_size=0.25,
                momentum_threshold=0.02
            ),
            'dual_model': DualModelStrategyConfig(
                q_buy_threshold=0.30,
                q_sell_threshold=0.20,
                ml_confidence_threshold=0.30,
                momentum_buy_threshold=1.5,
                ai_weight=0.7,
                momentum_weight=0.3
            )
        }
    
    def get_strategy_config(self, strategy_name: str, custom_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for a specific strategy with optional overrides"""
        
        # Get default config
        if strategy_name in self.default_configs:
            config = self.default_configs[strategy_name].to_dict()
        else:
            # Use base config as fallback
            config = BaseStrategyConfig().to_dict()
        
        # Apply custom overrides
        if custom_overrides:
            config.update(custom_overrides)
        
        # Load from config manager if available
        if self.config_manager:
            saved_config = self._load_from_config_manager(strategy_name)
            if saved_config:
                config.update(saved_config)
        
        return config
    
    def validate_config(self, strategy_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a configuration against its schema"""
        
        errors = []
        
        # Get appropriate schema
        schema_class = self.config_schemas.get(strategy_type, BaseStrategyConfig)
        
        # Check required fields
        required_fields = schema_class.__annotations__.keys()
        for field in required_fields:
            if field not in config and not hasattr(schema_class, field):
                errors.append(f"Missing required field: {field}")
        
        # Validate ranges
        if 'max_position_size' in config:
            if not 0 < config['max_position_size'] <= 1:
                errors.append("max_position_size must be between 0 and 1")
        
        if 'stop_loss_pct' in config:
            if not 0 < config['stop_loss_pct'] < 1:
                errors.append("stop_loss_pct must be between 0 and 1")
        
        if 'q_buy_threshold' in config and 'q_sell_threshold' in config:
            if config['q_buy_threshold'] <= config['q_sell_threshold']:
                errors.append("q_buy_threshold must be greater than q_sell_threshold")
        
        return len(errors) == 0, errors
    
    def save_config(self, strategy_name: str, config: Dict[str, Any], profile_name: str = None):
        """Save strategy configuration"""
        
        if not self.config_manager:
            return
        
        # Validate before saving
        strategy_type = self._get_strategy_type(strategy_name)
        valid, errors = self.validate_config(strategy_type, config)
        
        if not valid:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")
        
        # Save through config manager
        profile_data = {
            'strategy_name': strategy_name,
            'strategy_type': strategy_type,
            'config': config,
            'created': datetime.now().isoformat()
        }
        
        if profile_name:
            self.config_manager.save_profile(profile_name, profile_data)
        else:
            self._save_to_config_manager(strategy_name, config)
    
    def _get_strategy_type(self, strategy_name: str) -> str:
        """Determine strategy type from name"""
        
        if 'ai' in strategy_name.lower() and 'momentum' in strategy_name.lower():
            return 'dual_model'
        elif 'ai' in strategy_name.lower():
            return 'ai'
        elif 'momentum' in strategy_name.lower():
            return 'momentum'
        elif 'hybrid' in strategy_name.lower():
            return 'hybrid'
        else:
            return 'base'
    
    def _load_from_config_manager(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from config manager"""
        
        # Implementation depends on config_manager interface
        # This is a placeholder
        return None
    
    def _save_to_config_manager(self, strategy_name: str, config: Dict[str, Any]):
        """Save configuration to config manager"""
        
        # Implementation depends on config_manager interface
        # This is a placeholder
        pass
    
    def get_all_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all available strategy configurations"""
        
        configs = {}
        for name, config in self.default_configs.items():
            configs[name] = config.to_dict()
        
        return configs
    
    def export_config(self, strategy_name: str, filepath: str):
        """Export strategy configuration to JSON file"""
        
        config = self.get_strategy_config(strategy_name)
        
        export_data = {
            'strategy_name': strategy_name,
            'strategy_type': self._get_strategy_type(strategy_name),
            'config': config,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_config(self, filepath: str) -> Tuple[str, Dict[str, Any]]:
        """Import strategy configuration from JSON file"""
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        strategy_name = data.get('strategy_name')
        strategy_type = data.get('strategy_type', 'base')
        config = data.get('config', {})
        
        # Validate
        valid, errors = self.validate_config(strategy_type, config)
        if not valid:
            raise ValueError(f"Invalid configuration in {filepath}: {', '.join(errors)}")
        
        return strategy_name, config