"""
LAEF Daily Monitoring Setup Script
Sets up the daily monitoring system and creates necessary directories
"""

import os
import sys
import sqlite3
from datetime import datetime
import json

def create_directories():
    """Create necessary directories for LAEF monitoring"""
    directories = [
        "logs/daily_monitoring",
        "logs/knowledge", 
        "logs/training",
        "models",
        "data/cache",
        "config"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def setup_databases():
    """Setup SQLite databases for monitoring"""
    print("\n🗄️ Setting up databases...")
    
    # Knowledge database
    knowledge_db = "logs/knowledge/market_observations.db"
    conn = sqlite3.connect(knowledge_db)
    cursor = conn.cursor()
    
    # Market observations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            observation_type TEXT NOT NULL,
            symbol TEXT,
            data TEXT,
            confidence REAL DEFAULT 0.5,
            notes TEXT
        )
    ''')
    
    # Pattern observations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pattern_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            pattern_type TEXT NOT NULL,
            timeframe TEXT,
            symbols TEXT,
            pattern_data TEXT,
            outcome TEXT DEFAULT 'pending',
            accuracy_score REAL
        )
    ''')
    
    # Daily insights table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE DEFAULT (date('now')),
            insight_type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            supporting_data TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"  ✅ Knowledge database: {knowledge_db}")
    
    # Predictions database (from existing system)
    predictions_db = "logs/training/predictions.db"
    conn = sqlite3.connect(predictions_db)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
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
            prediction_accuracy TEXT DEFAULT 'pending',
            learning_applied BOOLEAN DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"  ✅ Predictions database: {predictions_db}")
    
    return True

def create_config_files():
    """Create configuration files"""
    print("\n⚙️ Creating configuration files...")
    
    # Main config
    config = {
        "daily_monitoring": {
            "wake_time": "08:00",
            "market_timezone": "US/Eastern",
            "watch_symbols": [
                "SPY", "QQQ", "IWM", "VIX",
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
                "XLF", "XLE", "XLK", "XLV", "XLI", "XLP"
            ],
            "prediction_intervals": {
                "micro": {"1min": 1, "5min": 5, "15min": 15},
                "short": {"30min": 30, "1hour": 60, "2hour": 120}, 
                "macro": {"4hour": 240, "1day": 1440}
            }
        },
        "knowledge_synthesis": {
            "min_observations_for_insight": 10,
            "confidence_threshold": 0.7,
            "pattern_success_threshold": 0.65
        },
        "pattern_recognition": {
            "micro_patterns": {
                "breakout": {"min_bars": 10, "threshold": 0.015},
                "reversal": {"min_bars": 5, "threshold": 0.01},
                "consolidation": {"min_bars": 15, "threshold": 0.005},
                "volume_spike": {"min_bars": 3, "threshold": 2.0}
            }
        }
    }
    
    config_file = "config/daily_monitoring_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  ✅ Config file: {config_file}")
    
    # Create example API keys file (user needs to fill this)
    api_keys_example = {
        "alpha_vantage_api_key": "YOUR_ALPHA_VANTAGE_KEY_HERE",
        "finnhub_api_key": "YOUR_FINNHUB_KEY_HERE", 
        "polygon_api_key": "YOUR_POLYGON_KEY_HERE",
        "note": "Replace with your actual API keys. Rename to api_keys.json"
    }
    
    api_keys_file = "config/api_keys_example.json"
    with open(api_keys_file, 'w') as f:
        json.dump(api_keys_example, f, indent=4)
    print(f"  ✅ API keys example: {api_keys_file}")
    
    return True

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    print("\n🚀 Creating startup scripts...")
    
    # Linux/Mac startup script
    linux_script = '''#!/bin/bash
# LAEF Daily Monitoring Startup Script

cd "$(dirname "$0")"

echo "Starting LAEF Daily Monitoring System..."
echo "LAEF will wake up at 8:00 AM EST every day"
echo "Press Ctrl+C to stop"

python daily_market_monitor.py

echo "LAEF Daily Monitoring stopped."
'''
    
    with open("start_laef_monitoring.sh", 'w') as f:
        f.write(linux_script)
    os.chmod("start_laef_monitoring.sh", 0o755)
    print("  ✅ Linux/Mac script: start_laef_monitoring.sh")
    
    # Windows startup script
    windows_script = '''@echo off
REM LAEF Daily Monitoring Startup Script

cd /d "%~dp0"

echo Starting LAEF Daily Monitoring System...
echo LAEF will wake up at 8:00 AM EST every day
echo Press Ctrl+C to stop

python daily_market_monitor.py

echo LAEF Daily Monitoring stopped.
pause
'''
    
    with open("start_laef_monitoring.bat", 'w') as f:
        f.write(windows_script)
    print("  ✅ Windows script: start_laef_monitoring.bat")
    
    return True

def create_requirements_file():
    """Create requirements.txt for dependencies"""
    print("\n📦 Creating requirements file...")
    
    requirements = '''# LAEF Daily Monitoring Dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
sqlite3  # Built into Python
schedule>=1.1.0
pytz>=2021.1
scikit-learn>=1.0.0
matplotlib>=3.4.0  # Optional for visualization
tensorflow>=2.8.0  # For AI components
yfinance>=0.1.70   # For market data
requests>=2.25.0
python-dateutil>=2.8.0

# Optional but recommended
jupyter>=1.0.0     # For analysis notebooks
plotly>=5.0.0      # For interactive charts
'''
    
    with open("requirements.txt", 'w') as f:
        f.write(requirements)
    print("  ✅ Requirements file: requirements.txt")
    
    return True

def create_readme():
    """Create README with setup instructions"""
    print("\n📖 Creating README...")
    
    readme_content = '''# LAEF Daily Market Monitoring System

Automated daily market observation and learning system for LAEF.

## Features

🌅 **Daily Wake-up**: LAEF wakes up at 8:00 AM EST every day
📊 **Continuous Monitoring**: Observes live markets during trading hours  
🧠 **Pattern Recognition**: Detects micro and macro patterns
📈 **Prediction Tracking**: Makes and tracks predictions across multiple timeframes
💡 **Knowledge Synthesis**: Learns from observations and improves over time
🚫 **No Trading**: Pure observation and learning mode

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Configuration**
   - Copy `config/api_keys_example.json` to `config/api_keys.json`
   - Add your actual API keys for market data

3. **Start Daily Monitoring**
   ```bash
   # Linux/Mac
   ./start_laef_monitoring.sh
   
   # Windows  
   start_laef_monitoring.bat
   
   # Or directly with Python
   python daily_market_monitor.py
   ```

4. **Control Interface**
   ```bash
   # Check status
   python laef_control.py status
   
   # View recent insights
   python laef_control.py insights
   
   # See all commands
   python laef_control.py help
   ```

## System Architecture

- **daily_market_monitor.py**: Main monitoring system
- **pattern_recognition_analyzer.py**: Multi-timeframe pattern detection
- **knowledge_synthesis.py**: Learning and insight generation
- **laef_control.py**: Command-line interface
- **prediction_tracker.py**: Tracks prediction accuracy

## Database Structure

- **Market Observations**: Real-time market observations and regime changes
- **Pattern Observations**: Detected patterns and their outcomes
- **Predictions**: Generated predictions and accuracy tracking
- **Daily Insights**: Synthesized learning insights

## Usage Examples

```bash
# Start the daily scheduler (runs continuously)
python laef_control.py scheduler

# View recent pattern analysis
python laef_control.py patterns 14

# Export all knowledge to JSON
python laef_control.py export

# Run manual knowledge synthesis
python laef_control.py synthesize
```

## Logs and Data

- `logs/daily_monitoring/`: Daily monitoring logs
- `logs/knowledge/`: Knowledge database and insights
- `logs/training/`: Prediction tracking data

## Customization

Edit `config/daily_monitoring_config.json` to customize:
- Watched symbols
- Prediction intervals  
- Pattern recognition parameters
- Wake-up time

## Notes

- System designed for observation and learning only
- No actual trading is performed
- Requires market data API keys for real-time data
- Works best with continuous operation for pattern learning

## Support

Run `python laef_control.py help` for available commands.
'''
    
    with open("README.md", 'w') as f:
        f.write(readme_content)
    print("  ✅ README: README.md")
    
    return True

def main():
    """Main setup function"""
    print("🤖 LAEF DAILY MONITORING SETUP")
    print("="*50)
    print("Setting up LAEF's automatic daily market monitoring system...")
    
    try:
        # Create directories
        create_directories()
        
        # Setup databases
        setup_databases()
        
        # Create config files
        create_config_files()
        
        # Create startup scripts
        create_startup_scripts()
        
        # Create requirements file
        create_requirements_file()
        
        # Create README
        create_readme()
        
        print("\n" + "="*50)
        print("✅ SETUP COMPLETE!")
        print("="*50)
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure API keys in config/api_keys.json")
        print("3. Start monitoring: python daily_market_monitor.py")
        print("4. Use control interface: python laef_control.py help")
        print("\n🎯 LAEF is now ready for daily market observation and learning!")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
