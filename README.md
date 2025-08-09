# LAEF Trading System

**Learn, Adapt, Execute, Forecast** - A fully autonomous, adaptive trading intelligence framework.

## 🚀 Quick Start

### Interactive Mode
```bash
python laef_unified_system.py
```

### Direct Launch
```bash
python start_laef_interactive.py
```

## 📁 Project Structure

### Core LAEF System
- **`laef/`** - Complete LAEF implementation with all 9 strategies
- **`laef_unified_system.py`** - Main entry point with interactive menu
- **`start_laef_interactive.py`** - Quick launcher script

### All 9 Trading Strategies ✅
1. **Momentum Scalping** - Quick buy/sell loops with ultra-tight stops
2. **Mean Reversion** - RSI/Bollinger Band reversals
3. **Statistical Arbitrage** - Pair trading with co-integration
4. **Dual-Model Swing** - Q-learning + ML profit prediction
5. **Pattern Recognition** - CNN/LSTM candlestick & chart patterns
6. **Time-Based Algorithm** - Session-specific behavior adjustments
7. **News Sentiment** - NLP analysis of company news & headlines
8. **Hybrid Adaptive** - Multi-strategy combination with dynamic weights
9. **Reinforced Grid Search** - Continuous parameter optimization

### Supporting Systems
- **`core/`** - Technical indicators, portfolio management
- **`data/`** - Multi-source data fetching and caching
- **`trading/`** - Alpaca API integration and backtesting
- **`training/`** - ML model training and prediction tracking
- **`optimization/`** - Strategy parameter optimization
- **`utils/`** - Utility functions and helpers
- **`config_profiles/`** - Pre-configured trading profiles
- **`models/`** - Trained ML models (Q-learning, etc.)

### Configuration & Data
- **`requirements.txt`** - Python dependencies
- **`config.py`** - System configuration
- **`.env`** - API keys and secrets (not in git)
- **`tickers_cleaned.csv`** - Symbol universe

### Documentation & Testing
- **`docs/`** - Training guides and documentation
- **`tests/`** - Test suites for backtesting
- **`testing/`** - System validation tests
- **`LAEF_Implementation_Progress.md`** - Implementation status report

### Archive
- **`legacy/`** - Obsolete files (organized and documented)

## 🎯 Available Modes

### 1. Paper Trading (Simulation)
- Uses Alpaca Paper Trading API
- Simulated trades with virtual money
- Perfect for testing strategies
- No financial risk

### 2. Backtesting (Historical Analysis)
- Analyze strategy performance on historical data
- No actual trading
- Generate performance reports
- ML-focused analysis

### 3. Live Learning (AI Training)
- Real-time market analysis and learning
- ML prediction tracking
- No actual trading
- Improve AI models

### 4. Configuration & Status
- Check API connection
- View current configuration
- Test components

## 🔧 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create `.env` file with:
```
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
```

### 3. Run System
```bash
python laef_unified_system.py
```

## 📊 System Features

- **✅ All 9 LAEF strategies implemented**
- **✅ Real-time market data integration**
- **✅ Multi-broker support (Alpaca)**
- **✅ Paper trading for safe testing**
- **✅ Comprehensive backtesting**
- **✅ ML-based predictions**
- **✅ Risk management**
- **✅ Performance analytics**
- **✅ Configuration profiles**

## 🛡️ Safety Features

- **Paper Trading First** - Always test with virtual money
- **Risk Management** - Built-in position sizing and stop losses
- **API Rate Limiting** - Respects broker API limits
- **Error Handling** - Graceful failure recovery
- **Logging** - Comprehensive trade and system logging

## 📈 Performance Tracking

- Win rate and profit/loss tracking
- Strategy-specific performance metrics
- Real-time portfolio monitoring
- Historical performance analysis
- Risk-adjusted returns

## 🔄 Continuous Learning

The LAEF system continuously learns and adapts:
- Strategy performance tracking
- Parameter optimization
- Market regime detection
- Adaptive position sizing
- Reinforcement learning feedback

## ⚠️ Important Notes

- **Start with Paper Trading** to test strategies
- **Never share API keys**
- **Monitor system performance** regularly
- **Review logs** for any issues
- **Backup your configuration** before changes

## 📞 Support

- Check `logs/` directory for system logs
- Review `LAEF_Implementation_Progress.md` for technical details
- All obsolete files moved to `legacy/` folder
- Configuration examples in `config_profiles/`

---

*LAEF System - Fully implemented with all 9 strategies - Ready for trading!* 🚀