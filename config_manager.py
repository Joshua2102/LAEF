# === config_manager.py - Safe Configuration Management System ===

import os
import json
import copy
from datetime import datetime
from typing import Dict, Any, Optional
import config  # Import original config

class ConfigurationManager:
    """
    Safe configuration management system that:
    1. Keeps original config.py untouched
    2. Stores optimized settings separately
    3. Allows switching between profiles
    4. Logs config settings with trading results
    5. Provides custom config input
    """
    
    def __init__(self):
        self.config_profiles_file = "config_profiles.json"
        self.config_profiles_dir = "config_profiles"
        os.makedirs(self.config_profiles_dir, exist_ok=True)
        self.profiles = self._load_profiles()
        self.current_profile = "original"
        self.current_config = None
        
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration profiles from file and directory"""
        profiles = {}
        
        # Load main profiles file
        if os.path.exists(self.config_profiles_file):
            try:
                with open(self.config_profiles_file, 'r') as f:
                    profiles.update(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load config profiles: {e}")
        
        # Load named profiles from directory
        profiles.update(self._load_named_profiles())
        
        # Ensure original profile exists
        if "original" not in profiles:
            profiles["original"] = self._extract_original_config()
        
        # Add momentum scalping default profile
        if "momentum_scalping" not in profiles:
            profiles["momentum_scalping"] = self._create_momentum_scalping_profile()
            
        return profiles
    
    def _load_named_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load all named profiles from config_profiles directory"""
        named_profiles = {}
        
        if not os.path.exists(self.config_profiles_dir):
            return named_profiles
            
        try:
            for filename in os.listdir(self.config_profiles_dir):
                if filename.endswith('.json') and filename != 'config_profiles.json':
                    profile_name = filename[:-5]  # Remove .json extension
                    filepath = os.path.join(self.config_profiles_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            profile_data = json.load(f)
                            # Ensure it has the right structure
                            if "trading_thresholds" in profile_data:
                                named_profiles[profile_name] = profile_data
                    except Exception as e:
                        print(f"Warning: Could not load profile {filename}: {e}")
                        
        except Exception as e:
            print(f"Warning: Could not scan profiles directory: {e}")
            
        return named_profiles
    
    def _extract_original_config(self) -> Dict[str, Any]:
        """Extract current config settings as the 'original' profile"""
        return {
            "name": "Original Configuration",
            "description": "Default LAEF trading configuration",
            "created": datetime.now().isoformat(),
            "engine_info": {
                "engine_type": "dual_model_engine",
                "engine_name": "LAEF Dual-Model Trading Engine",
                "engine_description": "Q-Learning + ML confidence dual model system",
                "engine_version": "2.0"
            },
            "trading_thresholds": {
                "q_buy": config.TRADING_THRESHOLDS.get('q_buy', 0.58),
                "q_sell": config.TRADING_THRESHOLDS.get('q_sell', 0.42),
                "ml_profit_peak": config.TRADING_THRESHOLDS.get('ml_profit_peak', 0.58),
                "rsi_oversold": config.TRADING_THRESHOLDS.get('rsi_oversold', 35),
                "rsi_overbought": config.TRADING_THRESHOLDS.get('rsi_overbought', 65),
                "sell_profit_pct": config.TRADING_THRESHOLDS.get('sell_profit_pct', 0.05),
                "stop_loss_pct": config.TRADING_THRESHOLDS.get('stop_loss_pct', 0.97),
                "trailing_stop_pct": config.TRADING_THRESHOLDS.get('trailing_stop_pct', 0.06)
            },
            "risk_management": {
                "initial_cash": config.INITIAL_CASH,
                "max_risk_per_trade": config.MAX_RISK_PER_TRADE,
                "max_position_size": config.MAX_POSITION_SIZE,
                "cooldown_minutes": config.COOLDOWN_MINUTES
            },
            "backtest_settings": {
                "start_date": config.BACKTEST_START_DATE,
                "end_date": config.BACKTEST_END_DATE
            }
        }
    
    def _create_momentum_scalping_profile(self) -> Dict[str, Any]:
        """Create default momentum scalping profile"""
        return {
            "name": "Momentum Scalping Configuration",
            "description": "Ultra-fast momentum-based micro-scalping engine optimized for 1-60 minute trades",
            "created": datetime.now().isoformat(),
            "engine_info": {
                "engine_type": "momentum_scalping_engine",
                "engine_name": "LAEF Momentum Scalping Engine",
                "engine_description": "Momentum-based micro-scalping with ultra-fast entry/exit",
                "engine_version": "1.0"
            },
            "trading_thresholds": {
                # Momentum-specific thresholds
                "momentum_buy_threshold": 2.0,
                "momentum_sell_threshold": -1.0,
                "momentum_acceleration_min": 0.5,
                "micro_profit_target": 0.15,
                "micro_stop_loss": 0.08,
                "trailing_micro_stop": 0.05,
                "volume_confirmation": 1.5,
                "volume_spike_threshold": 2.0,
                "max_hold_minutes": 30,
                "cooldown_seconds": 60,
                "scalp_position_pct": 0.02,
                "max_scalp_positions": 3,
                # Traditional indicators (still used for some confirmation)
                "rsi_oversold": 25,
                "rsi_overbought": 75,
            },
            "risk_management": {
                "initial_cash": config.INITIAL_CASH,
                "max_risk_per_trade": 0.02,  # Lower risk for scalping
                "max_position_size": 0.06,   # Smaller positions
                "cooldown_minutes": 1        # Quick cooldown
            },
            "backtest_settings": {
                "start_date": config.BACKTEST_START_DATE,
                "end_date": config.BACKTEST_END_DATE
            }
        }
    
    def save_profiles(self):
        """Save profiles to file"""
        try:
            with open(self.config_profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config profiles: {e}")
    
    def save_named_profile(self, profile_name: str, config_dict: Dict[str, Any]):
        """Save a configuration profile with a custom name"""
        # Sanitize profile name
        safe_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        if not safe_name:
            safe_name = f"custom_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to named file in profiles directory
        profile_file = os.path.join(self.config_profiles_dir, f"{safe_name}.json")
        
        try:
            with open(profile_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Also add to current profiles dict
            self.profiles[safe_name] = config_dict
            
            print(f"Configuration saved as '{safe_name}' in {profile_file}")
            return safe_name
            
        except Exception as e:
            print(f"Error saving profile '{profile_name}': {e}")
            return None
    
    def add_optimized_profile(self, optimized_config: Dict[str, Any], performance_data: Dict[str, Any] = None, engine_info: Dict[str, Any] = None):
        """Add or update the optimized configuration profile"""
        default_engine_info = {
            "engine_type": "dual_model_engine",
            "engine_name": "LAEF Dual-Model Trading Engine (Optimized)",
            "engine_description": "Optimized Q-Learning + ML confidence dual model system",
            "engine_version": "2.0"
        }
        
        self.profiles["optimized"] = {
            "name": "Profit-Optimized Configuration",
            "description": "Configuration optimized for maximum profit based on backtest analysis",
            "created": datetime.now().isoformat(),
            "engine_info": engine_info or default_engine_info,
            "trading_thresholds": {
                "q_buy": optimized_config.get('q_buy', 0.58),
                "q_sell": optimized_config.get('q_sell', 0.42),
                "ml_profit_peak": optimized_config.get('ml_profit_peak', 0.58),
                "rsi_oversold": optimized_config.get('rsi_oversold', 35),
                "rsi_overbought": optimized_config.get('rsi_overbought', 65),
                "sell_profit_pct": optimized_config.get('profit_target', 0.05),
                "stop_loss_pct": optimized_config.get('stop_loss', 0.97),
                "trailing_stop_pct": optimized_config.get('trailing_stop_pct', 0.06)
            },
            "risk_management": {
                "initial_cash": optimized_config.get('initial_cash', config.INITIAL_CASH),
                "max_risk_per_trade": optimized_config.get('risk_per_trade', config.MAX_RISK_PER_TRADE),
                "max_position_size": optimized_config.get('max_position', config.MAX_POSITION_SIZE),
                "cooldown_minutes": config.COOLDOWN_MINUTES
            },
            "backtest_settings": {
                "start_date": optimized_config.get('start_date', config.BACKTEST_START_DATE),
                "end_date": optimized_config.get('end_date', config.BACKTEST_END_DATE)
            },
            "performance_data": performance_data or {}
        }
        self.save_profiles()
    
    def select_configuration_interactive(self) -> Dict[str, Any]:
        """Interactive configuration selection with preview"""
        
        # Check if stdin is available for interactive input
        import sys
        if not sys.stdin or not sys.stdin.isatty():
            print("Non-interactive environment detected. Using original configuration.")
            if "original" in self.profiles:
                config_dict = self.profiles["original"]
                self.current_profile = "original"
                self.current_config = self._flatten_config(config_dict)
                return self.current_config
            else:
                raise Exception("No original configuration available")
        
        print("\n" + "=" * 60)
        print("CONFIGURATION SELECTION")
        print("=" * 60)
        
        # Reload profiles to get latest saved ones
        self.profiles = self._load_profiles()
        
        # Show available profiles
        print("\nAvailable Configuration Profiles:")
        
        profile_options = []
        option_num = 1
        
        # LAEF Default (Original) profile
        if "original" in self.profiles and self.profiles["original"]:
            print(f"{option_num}. LAEF DEFAULT - {self.profiles['original']['name']}")
            print(f"   {self.profiles['original']['description']}")
            profile_options.append("original")
            option_num += 1
        
        # Optimized profile
        if "optimized" in self.profiles and self.profiles["optimized"]:
            print(f"{option_num}. OPTIMIZED - {self.profiles['optimized']['name']}")
            print(f"   {self.profiles['optimized']['description']}")
            if "performance_data" in self.profiles["optimized"]:
                perf = self.profiles["optimized"]["performance_data"]
                if "expected_improvement" in perf:
                    print(f"   Expected improvement: {perf['expected_improvement']}")
            profile_options.append("optimized")
            option_num += 1
        
        # All other profiles (including legacy 'custom' and any named profiles)
        other_profiles = [p for p in self.profiles.keys() 
                         if p not in ["original", "optimized"] and self.profiles[p] is not None]
        
        if other_profiles:
            print(f"\nSaved Profiles:")
            for profile_key in sorted(other_profiles):
                profile_data = self.profiles[profile_key]
                display_name = profile_key.upper() if profile_key != "custom" else "CUSTOM"
                print(f"{option_num}. {display_name} - {profile_data.get('name', 'Named Profile')}")
                print(f"   {profile_data.get('description', 'Custom saved configuration')}")
                if "created" in profile_data:
                    try:
                        created = datetime.fromisoformat(profile_data["created"]).strftime("%Y-%m-%d %H:%M")
                        print(f"   Created: {created}")
                    except:
                        print(f"   Created: {profile_data['created']}")
                profile_options.append(profile_key)
                option_num += 1
        
        # Create new custom option
        print(f"\n{option_num}. CREATE NEW - Make a new custom configuration")
        profile_options.append("create_new")
        
        # Get selection
        while True:
            try:
                # Add timeout to prevent hanging
                try:
                    choice = input(f"\nSelect configuration (1-{len(profile_options)}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nInput interrupted. Using original configuration as fallback.")
                    if "original" in profile_options:
                        selected_profile = "original"
                        break
                    else:
                        raise Exception("No fallback configuration available")
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(profile_options):
                    selected_profile = profile_options[choice_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        # Handle selection
        if selected_profile == "create_new":
            return self._create_named_custom_config()
        else:
            config_dict = self.profiles[selected_profile]
            self._show_configuration_preview(config_dict, selected_profile.upper())
            
            try:
                confirm = input("\nUse this configuration? (Y/n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nInput interrupted. Using selected configuration as default.")
                confirm = 'y'
            
            if confirm in ['', 'y', 'yes']:
                self.current_profile = selected_profile
                self.current_config = self._flatten_config(config_dict)
                return self.current_config
            else:
                print("Configuration cancelled.")
                return self.select_configuration_interactive()
    
    def _show_configuration_preview(self, config_dict: Dict[str, Any], profile_name: str):
        """Show detailed configuration preview"""
        print(f"\n{profile_name} CONFIGURATION PREVIEW:")
        print("-" * 50)
        
        # Engine information
        engine_info = config_dict.get("engine_info", {})
        if engine_info:
            print("Trading Engine:")
            print(f"   • Engine: {engine_info.get('engine_name', 'Unknown')}")
            print(f"   • Type: {engine_info.get('engine_type', 'Unknown')}")
            print(f"   • Description: {engine_info.get('engine_description', 'N/A')}")
            print(f"   • Version: {engine_info.get('engine_version', 'N/A')}")
        
        # Trading thresholds - detect if this is microscalping
        thresholds = config_dict.get("trading_thresholds", {})
        is_microscalping = engine_info.get('engine_type') == 'ai_momentum_scalping_engine'
        
        if is_microscalping:
            print("\nAI Momentum Scalping Thresholds:")
            print(f"   • Q-Value Buy: {thresholds.get('q_buy', 'N/A')}")
            print(f"   • Q-Value Sell: {thresholds.get('q_sell', 'N/A')}")
            print(f"   • ML Profit Peak: {thresholds.get('ml_profit_peak', 'N/A')}")
            print(f"   • RSI Oversold: {thresholds.get('rsi_oversold', 'N/A')}")
            print(f"   • RSI Overbought: {thresholds.get('rsi_overbought', 'N/A')}")
            
            # Show microscalping-specific parameters
            print(f"   • Micro Profit Target: {thresholds.get('micro_profit_target', 0.12):.2f}%")
            print(f"   • Micro Stop Loss: {thresholds.get('micro_stop_loss', 0.08):.2f}%")
            print(f"   • Trailing Micro Stop: {thresholds.get('trailing_micro_stop', 0.05):.2f}%")
            
            print(f"\nMomentum Settings:")
            print(f"   • Momentum Buy Threshold: {thresholds.get('momentum_buy_threshold', 2.0)}")
            print(f"   • Volume Confirmation: {thresholds.get('volume_confirmation', 1.5):.1f}x")
            print(f"   • Max Hold Time: {thresholds.get('max_hold_minutes', 30)} minutes")
        else:
            print("\nDual-Model Trading Thresholds:")
            print(f"   • Q-Value Buy (Entry): {thresholds.get('q_buy', 'N/A')}")
            print(f"   • Q-Value Sell (Exit): {thresholds.get('q_sell', 'N/A')}")
            print(f"   • ML Profit Peak: {thresholds.get('ml_profit_peak', 'N/A')}")
            print(f"   • RSI Oversold: {thresholds.get('rsi_oversold', 'N/A')}")
            print(f"   • RSI Overbought: {thresholds.get('rsi_overbought', 'N/A')}")
            print(f"   • Profit Target: {thresholds.get('sell_profit_pct', 0)*100:.1f}%")
            print(f"   • Stop Loss: {(1-thresholds.get('stop_loss_pct', 0.97))*100:.1f}%")
            print(f"   • Trailing Stop: {thresholds.get('trailing_stop_pct', 0.06)*100:.1f}%")
        
        # Risk management
        risk = config_dict.get("risk_management", {})
        print(f"\nRisk Management:")
        print(f"   • Initial Cash: ${risk.get('initial_cash', 0):,}")
        print(f"   • Risk Per Trade: {risk.get('max_risk_per_trade', 0)*100:.1f}%")
        print(f"   • Max Position Size: {risk.get('max_position_size', 0)*100:.1f}%")
        if is_microscalping:
            print(f"   • Cooldown Time: {risk.get('cooldown_minutes', 1)} minutes")
        
        # Show creation date
        if "created" in config_dict:
            created_date = datetime.fromisoformat(config_dict["created"]).strftime("%Y-%m-%d %H:%M")
            print(f"\nCreated: {created_date}")
    
    def _create_named_custom_config(self) -> Dict[str, Any]:
        """Create custom configuration with user-provided name"""
        print("\nCREATE NEW CONFIGURATION PROFILE:")
        print("-" * 40)
        
        # Get profile name
        while True:
            profile_name = input("Enter a name for this configuration profile: ").strip()
            if profile_name:
                # Check if name already exists
                if profile_name.lower().replace(' ', '_') in self.profiles:
                    overwrite = input(f"Profile '{profile_name}' already exists. Overwrite? (y/N): ").strip().lower()
                    if overwrite in ['y', 'yes']:
                        break
                    else:
                        continue
                else:
                    break
            else:
                print("Please enter a valid profile name.")
        
        # Get optional description
        description = input("Enter a description (optional): ").strip()
        if not description:
            description = f"Custom trading configuration created on {datetime.now().strftime('%Y-%m-%d')}"
        
        return self._create_custom_config_with_name(profile_name, description)
    
    def _create_custom_config_with_name(self, profile_name: str, description: str) -> Dict[str, Any]:
        """Create custom configuration with specified name and description"""
        print(f"\nSETTING UP CONFIGURATION: {profile_name}")
        print("-" * 50)
        
        # Detect if this is a microscalping/scalping profile
        is_microscalping = self._is_microscalping_profile(profile_name)
        
        if is_microscalping:
            return self._create_microscalping_config(profile_name, description)
        else:
            return self._create_standard_config(profile_name, description)
    
    def _is_microscalping_profile(self, profile_name: str) -> bool:
        """Detect if profile name indicates microscalping strategy"""
        scalping_keywords = [
            'microscalping', 'micro-scalping', 'scalping', 'scalp',
            'micro', 'fast', 'quick', 'ultra', 'speed', 'momentum'
        ]
        profile_lower = profile_name.lower()
        return any(keyword in profile_lower for keyword in scalping_keywords)
    
    def _create_microscalping_config(self, profile_name: str, description: str) -> Dict[str, Any]:
        """Create microscalping-specific configuration with appropriate ranges"""
        print(f"\n🚀 MICROSCALPING PROFILE DETECTED")
        print("Optimized for AI Momentum Scalping Engine")
        print("Using micro-profit ranges for frequent, fast trades\n")
        
        # Start with AI momentum scalping base
        if "momentum_scalping" in self.profiles:
            base_config = self.profiles["momentum_scalping"]
        else:
            base_config = self._create_momentum_scalping_profile()
        
        custom_config = copy.deepcopy(base_config)
        custom_config["name"] = profile_name
        custom_config["description"] = description
        custom_config["created"] = datetime.now().isoformat()
        
        # Set AI Momentum Scalping engine info
        custom_config["engine_info"] = {
            "engine_type": "ai_momentum_scalping_engine",
            "engine_name": "LAEF AI Momentum Scalping Engine",
            "engine_description": "AI-powered micro-scalping with momentum analysis",
            "engine_version": "1.0"
        }
        
        print("Enter new values (press Enter to keep current value):")
        print("Note: All ranges are optimized for microscalping strategy\n")
        
        # Trading thresholds with microscalping ranges
        thresholds = custom_config["trading_thresholds"]
        print("AI Momentum Scalping Thresholds:")
        
        # Q-values (scalping appropriate ranges)
        new_val = input(f"   Q-Value Buy (0.15-0.45) [{thresholds.get('q_buy', 0.25)}]: ").strip()
        if new_val: 
            val = max(0.15, min(0.45, float(new_val)))
            thresholds['q_buy'] = val
            if val != float(new_val):
                print(f"     → Adjusted to {val} (scalping range)")
        
        new_val = input(f"   Q-Value Sell (0.05-0.25) [{thresholds.get('q_sell', 0.15)}]: ").strip()
        if new_val: 
            val = max(0.05, min(0.25, float(new_val)))
            thresholds['q_sell'] = val
            if val != float(new_val):
                print(f"     → Adjusted to {val} (scalping range)")
        
        new_val = input(f"   ML Profit Peak (0.15-0.50) [{thresholds.get('ml_profit_peak', 0.25)}]: ").strip()
        if new_val: 
            val = max(0.15, min(0.50, float(new_val)))
            thresholds['ml_profit_peak'] = val
            if val != float(new_val):
                print(f"     → Adjusted to {val} (scalping range)")
        
        # RSI (scalping optimized)
        new_val = input(f"   RSI Oversold (15-35) [{thresholds.get('rsi_oversold', 25)}]: ").strip()
        if new_val: 
            val = max(15, min(35, int(new_val)))
            thresholds['rsi_oversold'] = val
            if val != int(new_val):
                print(f"     → Adjusted to {val} (scalping range)")
        
        new_val = input(f"   RSI Overbought (65-85) [{thresholds.get('rsi_overbought', 75)}]: ").strip()
        if new_val: 
            val = max(65, min(85, int(new_val)))
            thresholds['rsi_overbought'] = val
            if val != int(new_val):
                print(f"     → Adjusted to {val} (scalping range)")
        
        # Micro profit targets (basis points)
        current_profit = thresholds.get('micro_profit_target', 0.12)
        new_val = input(f"   Micro Profit Target (0.05-0.50%) [{current_profit}]: ").strip()
        if new_val: 
            val = max(0.05, min(0.50, float(new_val)))
            thresholds['micro_profit_target'] = val
            # Also set the regular profit target field for compatibility
            thresholds['sell_profit_pct'] = val / 100
            if val != float(new_val):
                print(f"     → Adjusted to {val}% (scalping range)")
        
        # Micro stop loss (basis points)
        current_stop = thresholds.get('micro_stop_loss', 0.08)
        new_val = input(f"   Micro Stop Loss (0.03-0.20%) [{current_stop}]: ").strip()
        if new_val: 
            val = max(0.03, min(0.20, float(new_val)))
            thresholds['micro_stop_loss'] = val
            # Also set the regular stop loss field for compatibility
            thresholds['stop_loss_pct'] = 1 - (val / 100)
            if val != float(new_val):
                print(f"     → Adjusted to {val}% (scalping range)")
        
        # Trailing micro stop
        current_trailing = thresholds.get('trailing_micro_stop', 0.05)
        new_val = input(f"   Trailing Micro Stop (0.02-0.10%) [{current_trailing}]: ").strip()
        if new_val: 
            val = max(0.02, min(0.10, float(new_val)))
            thresholds['trailing_micro_stop'] = val
            thresholds['trailing_stop_pct'] = val / 100
            if val != float(new_val):
                print(f"     → Adjusted to {val}% (scalping range)")
        
        # Momentum settings
        print("\nMomentum Scalping Settings:")
        
        current_momentum = thresholds.get('momentum_buy_threshold', 2.0)
        new_val = input(f"   Momentum Buy Threshold (1.0-5.0) [{current_momentum}]: ").strip()
        if new_val: 
            val = max(1.0, min(5.0, float(new_val)))
            thresholds['momentum_buy_threshold'] = val
            if val != float(new_val):
                print(f"     → Adjusted to {val} (scalping range)")
        
        current_volume = thresholds.get('volume_confirmation', 1.5)
        new_val = input(f"   Volume Confirmation (1.2-3.0x) [{current_volume}]: ").strip()
        if new_val: 
            val = max(1.2, min(3.0, float(new_val)))
            thresholds['volume_confirmation'] = val
            if val != float(new_val):
                print(f"     → Adjusted to {val}x (scalping range)")
        
        current_hold = thresholds.get('max_hold_minutes', 30)
        new_val = input(f"   Max Hold Time (5-120 minutes) [{current_hold}]: ").strip()
        if new_val: 
            val = max(5, min(120, int(new_val)))
            thresholds['max_hold_minutes'] = val
            if val != int(new_val):
                print(f"     → Adjusted to {val} minutes (scalping range)")
        
        # Risk management with scalping-appropriate ranges
        self._configure_microscalping_risk_management(custom_config)
        
        return self._finalize_custom_config(custom_config, profile_name)
        
    def _create_standard_config(self, profile_name: str, description: str) -> Dict[str, Any]:
        """Create standard dual-model configuration (original logic)"""
        # Start with original as base
        base_config = self.profiles["original"]
        custom_config = copy.deepcopy(base_config)
        custom_config["name"] = profile_name
        custom_config["description"] = description
        custom_config["created"] = datetime.now().isoformat()
        
        # Ensure engine info is present
        if "engine_info" not in custom_config:
            custom_config["engine_info"] = {
                "engine_type": "dual_model_engine",
                "engine_name": "LAEF Dual-Model Trading Engine",
                "engine_description": "Q-Learning + ML confidence dual model system",
                "engine_version": "2.0"
            }
        
        print("Enter new values (press Enter to keep current value):")
        
        # Trading thresholds
        thresholds = custom_config["trading_thresholds"]
        print("\nDual-Model Trading Thresholds:")
        
        new_val = input(f"   Q-Value Buy [{thresholds['q_buy']}]: ").strip()
        if new_val: thresholds['q_buy'] = float(new_val)
        
        new_val = input(f"   Q-Value Sell [{thresholds['q_sell']}]: ").strip()
        if new_val: thresholds['q_sell'] = float(new_val)
        
        new_val = input(f"   ML Profit Peak [{thresholds['ml_profit_peak']}]: ").strip()
        if new_val: thresholds['ml_profit_peak'] = float(new_val)
        
        new_val = input(f"   RSI Oversold [{thresholds['rsi_oversold']}]: ").strip()
        if new_val: thresholds['rsi_oversold'] = int(new_val)
        
        new_val = input(f"   RSI Overbought [{thresholds['rsi_overbought']}]: ").strip()
        if new_val: thresholds['rsi_overbought'] = int(new_val)
        
        new_val = input(f"   Profit Target % [{thresholds['sell_profit_pct']*100:.1f}]: ").strip()
        if new_val: thresholds['sell_profit_pct'] = float(new_val) / 100
        
        new_val = input(f"   Stop Loss % [{(1-thresholds['stop_loss_pct'])*100:.1f}]: ").strip()
        if new_val: thresholds['stop_loss_pct'] = 1 - (float(new_val) / 100)
        
        # Get trailing stop percentage (default to 6% if not present)
        trailing_stop_default = thresholds.get('trailing_stop_pct', 0.06)
        new_val = input(f"   Trailing Stop % [{trailing_stop_default*100:.1f}]: ").strip()
        if new_val: 
            thresholds['trailing_stop_pct'] = float(new_val) / 100
        else:
            thresholds['trailing_stop_pct'] = trailing_stop_default
        
        # Risk management
        risk = custom_config["risk_management"]
        print("\nRisk Management:")
        
        new_val = input(f"   Initial Cash [{risk['initial_cash']}]: ").strip()
        if new_val: risk['initial_cash'] = float(new_val)
        
        new_val = input(f"   Risk Per Trade % [{risk['max_risk_per_trade']*100:.1f}]: ").strip()
        if new_val: risk['max_risk_per_trade'] = float(new_val) / 100
        
        new_val = input(f"   Max Position Size % [{risk['max_position_size']*100:.1f}]: ").strip()
        if new_val: risk['max_position_size'] = float(new_val) / 100
        
        return self._finalize_custom_config(custom_config, profile_name)
    
    def _configure_microscalping_risk_management(self, custom_config: Dict[str, Any]):
        """Configure risk management with microscalping-appropriate ranges"""
        risk = custom_config["risk_management"]
        print("\nMicroscalping Risk Management:")
        
        new_val = input(f"   Initial Cash [{risk['initial_cash']}]: ").strip()
        if new_val: risk['initial_cash'] = float(new_val)
        
        # Risk per trade (lower for frequent scalping)
        current_risk = risk.get('max_risk_per_trade', 0.02)
        new_val = input(f"   Risk Per Trade % (1-5%) [{current_risk*100:.1f}]: ").strip()
        if new_val: 
            val = max(0.01, min(0.05, float(new_val) / 100))
            risk['max_risk_per_trade'] = val
            if val != float(new_val) / 100:
                print(f"     → Adjusted to {val*100:.1f}% (scalping range)")
        
        # Position size (smaller for frequent trading)
        current_position = risk.get('max_position_size', 0.06)
        new_val = input(f"   Max Position Size % (1-10%) [{current_position*100:.1f}]: ").strip()
        if new_val: 
            val = max(0.01, min(0.10, float(new_val) / 100))
            risk['max_position_size'] = val
            if val != float(new_val) / 100:
                print(f"     → Adjusted to {val*100:.1f}% (scalping range)")
        
        # Cooldown (short for scalping)
        current_cooldown = risk.get('cooldown_minutes', 1)
        new_val = input(f"   Cooldown Minutes (0.5-5) [{current_cooldown}]: ").strip()
        if new_val: 
            val = max(0.5, min(5.0, float(new_val)))
            risk['cooldown_minutes'] = val
            if val != float(new_val):
                print(f"     → Adjusted to {val} minutes (scalping range)")
    
    def _finalize_custom_config(self, custom_config: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        """Show preview and confirm configuration"""
        # Show preview and confirm
        self._show_configuration_preview(custom_config, profile_name.upper())
        
        confirm = input(f"\nSave and use '{profile_name}' configuration? (Y/n): ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            # Save named profile
            saved_name = self.save_named_profile(profile_name, custom_config)
            if saved_name:
                self.current_profile = saved_name
                self.current_config = self._flatten_config(custom_config)
                return self.current_config
            else:
                print("Failed to save configuration.")
                return self.select_configuration_interactive()
        else:
            print("Configuration cancelled.")
            return self.select_configuration_interactive()
    
    def _create_custom_config(self) -> Dict[str, Any]:
        """Legacy method - redirects to named config creation"""
        return self._create_named_custom_config()
    
    def _flatten_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten config structure for easy use in trading code"""
        # Upgrade legacy profile if needed
        config_dict = self._upgrade_legacy_profile(config_dict)
        
        flat_config = {}
        
        # Trading thresholds
        if "trading_thresholds" in config_dict:
            flat_config.update(config_dict["trading_thresholds"])
        
        # Risk management
        if "risk_management" in config_dict:
            flat_config.update(config_dict["risk_management"])
        
        # Backtest settings
        if "backtest_settings" in config_dict:
            flat_config.update(config_dict["backtest_settings"])
        
        # Engine information
        if "engine_info" in config_dict:
            engine_info = config_dict["engine_info"]
            flat_config["engine_type"] = engine_info.get("engine_type", "dual_model_engine")
            flat_config["engine_name"] = engine_info.get("engine_name", "LAEF Dual-Model Trading Engine")
            flat_config["engine_description"] = engine_info.get("engine_description", "Q-Learning + ML confidence system")
            flat_config["engine_version"] = engine_info.get("engine_version", "2.0")
        
        # Add profile metadata
        flat_config["_profile_name"] = config_dict.get("name", "Unknown")
        flat_config["_profile_created"] = config_dict.get("created", "Unknown")
        
        return flat_config
    
    def _upgrade_legacy_profile(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add engine_info to legacy profiles that don't have it"""
        if "engine_info" not in config_dict:
            config_dict = copy.deepcopy(config_dict)
            config_dict["engine_info"] = {
                "engine_type": "dual_model_engine",
                "engine_name": "LAEF Dual-Model Trading Engine",
                "engine_description": "Q-Learning + ML confidence dual model system",
                "engine_version": "2.0"
            }
        return config_dict
    
    def get_config_for_logging(self) -> Dict[str, Any]:
        """Get current config settings for logging with trading results"""
        if not self.current_config:
            self.current_config = self._flatten_config(self.profiles["original"])
        
        return {
            "profile_used": self.current_profile,
            "profile_name": self.current_config.get("_profile_name", "Unknown"),
            "created": self.current_config.get("_profile_created", "Unknown"),
            "settings": {
                "q_buy": self.current_config.get("q_buy"),
                "q_sell": self.current_config.get("q_sell"), 
                "ml_profit_peak": self.current_config.get("ml_profit_peak"),
                "rsi_oversold": self.current_config.get("rsi_oversold"),
                "rsi_overbought": self.current_config.get("rsi_overbought"),
                "sell_profit_pct": self.current_config.get("sell_profit_pct"),
                "stop_loss_pct": self.current_config.get("stop_loss_pct"),
                "trailing_stop_pct": self.current_config.get("trailing_stop_pct"),
                "max_risk_per_trade": self.current_config.get("max_risk_per_trade"),
                "max_position_size": self.current_config.get("max_position_size"),
                "initial_cash": self.current_config.get("initial_cash")
            }
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the currently selected configuration"""
        if not self.current_config:
            # Default to original if none selected
            self.current_config = self._flatten_config(self.profiles["original"])
            self.current_profile = "original"
        
        return self.current_config
    
    def list_profiles(self):
        """List all available configuration profiles"""
        print("\nAvailable Configuration Profiles:")
        print("=" * 40)
        
        for profile_key, profile_data in self.profiles.items():
            if profile_data:
                print(f"\n{profile_key.upper()}:")
                print(f"   Name: {profile_data.get('name', 'Unknown')}")
                print(f"   Description: {profile_data.get('description', 'No description')}")
                if "created" in profile_data:
                    created = datetime.fromisoformat(profile_data["created"]).strftime("%Y-%m-%d %H:%M")
                    print(f"   Created: {created}")

# Global instance
config_manager = ConfigurationManager()