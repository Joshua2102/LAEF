#!/usr/bin/env python3
"""
LAEF Smart Stock Selection System
Intelligently selects stocks for trading based on multiple criteria
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
import warnings
warnings.filterwarnings('ignore')

class SmartStockSelector:
    """Intelligent stock selection based on multiple criteria and ML predictions"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        # Get API credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found")
        
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Selection criteria weights
        self.criteria_weights = {
            'momentum': 0.25,
            'volatility': 0.20,
            'volume': 0.15,
            'trend': 0.20,
            'ml_score': 0.20
        }
        
        # Universe of stocks to consider
        self.stock_universe = self._get_default_universe()
        
    def _get_default_universe(self) -> List[str]:
        """Get default universe of stocks to analyze"""
        # Tech giants
        tech = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC']
        
        # Financial
        financial = ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C']
        
        # Consumer
        consumer = ['AMZN', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX']
        
        # Healthcare
        healthcare = ['JNJ', 'PFE', 'UNH', 'CVS', 'ABBV']
        
        # Energy
        energy = ['XOM', 'CVX', 'COP']
        
        # ETFs for market breadth
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA']
        
        return tech + financial + consumer + healthcare + energy + etfs
    
    def select_stocks(self, 
                     num_stocks: int = 5,
                     selection_mode: str = 'balanced',
                     custom_universe: List[str] = None,
                     lookback_days: int = 30) -> Dict:
        """
        Select best stocks for trading
        
        Args:
            num_stocks: Number of stocks to select
            selection_mode: 'momentum', 'value', 'volatility', 'balanced', 'ml_driven'
            custom_universe: Custom list of stocks to analyze
            lookback_days: Days of history to analyze
            
        Returns:
            Dictionary with selected stocks and analysis
        """
        print(f"Starting smart stock selection (mode: {selection_mode})...")
        
        # Use custom universe if provided
        universe = custom_universe or self.stock_universe
        
        # Adjust weights based on selection mode
        self._adjust_weights_for_mode(selection_mode)
        
        # Analyze all stocks in universe
        stock_scores = {}
        stock_analysis = {}
        
        for symbol in universe:
            try:
                analysis = self._analyze_stock(symbol, lookback_days)
                if analysis:
                    score = self._calculate_stock_score(analysis, selection_mode)
                    stock_scores[symbol] = score
                    stock_analysis[symbol] = analysis
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        
        # Select top stocks
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = [stock[0] for stock in sorted_stocks[:num_stocks]]
        
        # Generate selection report
        report = self._generate_selection_report(
            selected_stocks, stock_scores, stock_analysis, selection_mode
        )
        
        return {
            'selected_stocks': selected_stocks,
            'scores': {s: stock_scores[s] for s in selected_stocks},
            'analysis': {s: stock_analysis[s] for s in selected_stocks},
            'report': report,
            'selection_criteria': self.criteria_weights,
            'timestamp': datetime.now()
        }
    
    def _adjust_weights_for_mode(self, mode: str):
        """Adjust selection criteria weights based on mode"""
        if mode == 'momentum':
            self.criteria_weights = {
                'momentum': 0.40,
                'volatility': 0.15,
                'volume': 0.15,
                'trend': 0.20,
                'ml_score': 0.10
            }
        elif mode == 'value':
            self.criteria_weights = {
                'momentum': 0.10,
                'volatility': 0.25,
                'volume': 0.20,
                'trend': 0.15,
                'ml_score': 0.30
            }
        elif mode == 'volatility':
            self.criteria_weights = {
                'momentum': 0.20,
                'volatility': 0.35,
                'volume': 0.20,
                'trend': 0.10,
                'ml_score': 0.15
            }
        elif mode == 'ml_driven':
            self.criteria_weights = {
                'momentum': 0.15,
                'volatility': 0.15,
                'volume': 0.10,
                'trend': 0.20,
                'ml_score': 0.40
            }
        else:  # balanced
            self.criteria_weights = {
                'momentum': 0.25,
                'volatility': 0.20,
                'volume': 0.15,
                'trend': 0.20,
                'ml_score': 0.20
            }
    
    def _analyze_stock(self, symbol: str, lookback_days: int) -> Optional[Dict]:
        """Comprehensive analysis of a single stock"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            
            if symbol not in bars.data or len(bars.data[symbol]) < 10:
                return None
            
            # Convert to DataFrame
            data = []
            for bar in bars.data[symbol]:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate metrics
            analysis = {
                'symbol': symbol,
                'current_price': df['close'].iloc[-1],
                'momentum_score': self._calculate_momentum(df),
                'volatility_score': self._calculate_volatility_score(df),
                'volume_score': self._calculate_volume_score(df),
                'trend_score': self._calculate_trend_score(df),
                'ml_score': self._calculate_ml_score(df),
                'technical_indicators': self._calculate_technical_indicators(df),
                'price_action': self._analyze_price_action(df),
                'risk_metrics': self._calculate_risk_metrics(df)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (0-100)"""
        # Multiple timeframe momentum
        returns_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) if len(df) >= 5 else 0
        returns_10d = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) if len(df) >= 10 else 0
        returns_20d = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) if len(df) >= 20 else 0
        
        # Rate of change
        roc = df['close'].pct_change().rolling(5).mean().iloc[-1]
        
        # Combine metrics
        momentum = (
            returns_5d * 0.4 +
            returns_10d * 0.3 +
            returns_20d * 0.2 +
            roc * 0.1
        )
        
        # Normalize to 0-100
        return max(0, min(100, 50 + momentum * 500))
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility score (0-100, higher = more suitable for trading)"""
        # Daily volatility
        daily_vol = df['close'].pct_change().std()
        
        # Intraday volatility (high-low range)
        avg_range = ((df['high'] - df['low']) / df['close']).mean()
        
        # ATR-like metric
        tr = pd.DataFrame()
        tr['hl'] = df['high'] - df['low']
        tr['hc'] = abs(df['high'] - df['close'].shift(1))
        tr['lc'] = abs(df['low'] - df['close'].shift(1))
        atr = tr.max(axis=1).rolling(14).mean().iloc[-1] / df['close'].iloc[-1]
        
        # Optimal volatility for trading (not too high, not too low)
        if daily_vol < 0.005:  # Too low
            vol_score = 30
        elif daily_vol > 0.05:  # Too high
            vol_score = 40
        else:  # Sweet spot
            vol_score = 70 + (1 - abs(daily_vol - 0.02) / 0.02) * 30
        
        return vol_score
    
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume score (0-100)"""
        # Volume trend
        recent_vol = df['volume'].iloc[-5:].mean()
        older_vol = df['volume'].iloc[-20:-5].mean() if len(df) > 20 else recent_vol
        
        vol_increase = (recent_vol / older_vol - 1) if older_vol > 0 else 0
        
        # Volume consistency
        vol_std = df['volume'].rolling(10).std().iloc[-1]
        vol_mean = df['volume'].rolling(10).mean().iloc[-1]
        vol_consistency = 1 - (vol_std / vol_mean) if vol_mean > 0 else 0.5
        
        # Combine metrics
        volume_score = (
            min(100, 50 + vol_increase * 100) * 0.6 +
            vol_consistency * 100 * 0.4
        )
        
        return max(0, min(100, volume_score))
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend score (0-100)"""
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # Trend alignment
        current_price = df['close'].iloc[-1]
        sma_10 = df['sma_10'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        
        # Bullish alignment
        if current_price > sma_10 > sma_20:
            alignment_score = 80
        elif current_price > sma_10 or current_price > sma_20:
            alignment_score = 60
        elif current_price < sma_10 < sma_20:
            alignment_score = 20
        else:
            alignment_score = 40
        
        # Trend strength (slope of moving average)
        if len(df) >= 10:
            ma_slope = (sma_10 - df['sma_10'].iloc[-10]) / df['sma_10'].iloc[-10]
            trend_strength = min(100, 50 + ma_slope * 1000)
        else:
            trend_strength = 50
        
        # Combine
        trend_score = alignment_score * 0.6 + trend_strength * 0.4
        
        return max(0, min(100, trend_score))
    
    def _calculate_ml_score(self, df: pd.DataFrame) -> float:
        """Calculate ML-based score using technical indicators"""
        # This is a simplified ML score based on technical patterns
        # In production, this would use the actual LAEF agent
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        macd_diff = (macd - signal).iloc[-1]
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        bb_position = (df['close'].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        # Combine into ML score
        ml_score = 50  # Base score
        
        # RSI contribution
        if rsi < 30:
            ml_score += 20
        elif rsi > 70:
            ml_score -= 10
        else:
            ml_score += (50 - abs(rsi - 50)) / 5
        
        # MACD contribution
        if macd_diff > 0:
            ml_score += min(15, macd_diff * 10)
        else:
            ml_score += max(-10, macd_diff * 5)
        
        # Bollinger Bands contribution
        if bb_position < 0.2:
            ml_score += 15
        elif bb_position > 0.8:
            ml_score -= 10
        else:
            ml_score += 5
        
        return max(0, min(100, ml_score))
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate additional technical indicators"""
        indicators = {}
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Moving averages
        indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
        
        # Price position
        min_price = df['low'].rolling(20).min().iloc[-1]
        max_price = df['high'].rolling(20).max().iloc[-1]
        indicators['price_position'] = (df['close'].iloc[-1] - min_price) / (max_price - min_price) if max_price > min_price else 0.5
        
        return indicators
    
    def _analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze recent price action patterns"""
        price_action = {}
        
        # Recent performance
        price_action['1d_return'] = df['close'].pct_change().iloc[-1]
        price_action['5d_return'] = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) if len(df) >= 5 else 0
        
        # Trend consistency
        recent_closes = df['close'].iloc[-5:]
        price_action['trend_consistency'] = (recent_closes.diff() > 0).sum() / 4 if len(recent_closes) >= 5 else 0.5
        
        # Support/Resistance levels
        price_action['near_support'] = df['close'].iloc[-1] <= df['low'].rolling(20).min().iloc[-1] * 1.02
        price_action['near_resistance'] = df['close'].iloc[-1] >= df['high'].rolling(20).max().iloc[-1] * 0.98
        
        return price_action
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk-related metrics"""
        risk_metrics = {}
        
        # Downside deviation
        returns = df['close'].pct_change()
        downside_returns = returns[returns < 0]
        risk_metrics['downside_deviation'] = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        risk_metrics['max_drawdown'] = drawdown.min()
        
        # Value at Risk (95%)
        risk_metrics['var_95'] = returns.quantile(0.05)
        
        return risk_metrics
    
    def _calculate_stock_score(self, analysis: Dict, mode: str) -> float:
        """Calculate overall stock score based on all factors"""
        scores = {
            'momentum': analysis['momentum_score'],
            'volatility': analysis['volatility_score'],
            'volume': analysis['volume_score'],
            'trend': analysis['trend_score'],
            'ml_score': analysis['ml_score']
        }
        
        # Apply mode-specific adjustments
        if mode == 'momentum':
            # Boost stocks with strong recent performance
            if analysis['price_action']['5d_return'] > 0.02:
                scores['momentum'] *= 1.2
        elif mode == 'value':
            # Boost oversold stocks
            if analysis['technical_indicators']['rsi'] < 35:
                scores['ml_score'] *= 1.3
        elif mode == 'volatility':
            # Extra weight for optimal volatility range
            if 0.01 < analysis['risk_metrics']['downside_deviation'] < 0.03:
                scores['volatility'] *= 1.2
        
        # Calculate weighted score
        total_score = sum(
            scores[criteria] * weight 
            for criteria, weight in self.criteria_weights.items()
        )
        
        # Apply risk adjustments
        if analysis['risk_metrics']['max_drawdown'] < -0.15:
            total_score *= 0.9  # Penalize high drawdown
        
        return total_score
    
    def _generate_selection_report(self, 
                                  selected_stocks: List[str],
                                  scores: Dict[str, float],
                                  analysis: Dict[str, Dict],
                                  mode: str) -> str:
        """Generate detailed selection report"""
        report = []
        report.append("=" * 60)
        report.append("SMART STOCK SELECTION REPORT")
        report.append(f"Selection Mode: {mode.upper()}")
        report.append(f"Generated: {datetime.now()}")
        report.append("=" * 60)
        
        report.append(f"\nSELECTED STOCKS ({len(selected_stocks)}):")
        report.append("-" * 40)
        
        for i, symbol in enumerate(selected_stocks):
            stock_analysis = analysis[symbol]
            stock_score = scores[symbol]
            
            report.append(f"\n{i+1}. {symbol} - Score: {stock_score:.2f}")
            report.append(f"   Current Price: ${stock_analysis['current_price']:.2f}")
            report.append(f"   Momentum: {stock_analysis['momentum_score']:.1f}/100")
            report.append(f"   Trend: {stock_analysis['trend_score']:.1f}/100")
            report.append(f"   ML Score: {stock_analysis['ml_score']:.1f}/100")
            report.append(f"   5-Day Return: {stock_analysis['price_action']['5d_return']:.2%}")
            report.append(f"   RSI: {stock_analysis['technical_indicators']['rsi']:.1f}")
            
            # Selection rationale
            rationale = self._generate_selection_rationale(stock_analysis, mode)
            report.append(f"   Why Selected: {rationale}")
        
        # Overall market assessment
        report.append("\n" + "=" * 40)
        report.append("MARKET ASSESSMENT:")
        report.append(self._generate_market_assessment(analysis))
        
        # Risk warnings
        report.append("\nRISK CONSIDERATIONS:")
        for symbol in selected_stocks:
            risk = analysis[symbol]['risk_metrics']
            if risk['max_drawdown'] < -0.10:
                report.append(f"  • {symbol}: Recent drawdown of {risk['max_drawdown']:.1%}")
        
        return "\n".join(report)
    
    def _generate_selection_rationale(self, analysis: Dict, mode: str) -> str:
        """Generate explanation for why a stock was selected"""
        reasons = []
        
        # Check each factor
        if analysis['momentum_score'] > 70:
            reasons.append("strong momentum")
        if analysis['trend_score'] > 75:
            reasons.append("solid uptrend")
        if analysis['ml_score'] > 70:
            reasons.append("positive ML signals")
        if analysis['volume_score'] > 65:
            reasons.append("increasing volume")
        
        # Mode-specific reasons
        if mode == 'momentum' and analysis['price_action']['5d_return'] > 0.03:
            reasons.append("exceptional recent performance")
        elif mode == 'value' and analysis['technical_indicators']['rsi'] < 35:
            reasons.append("oversold conditions")
        elif mode == 'volatility' and 60 < analysis['volatility_score'] < 80:
            reasons.append("optimal volatility for trading")
        
        return ", ".join(reasons) if reasons else "balanced technical profile"
    
    def _generate_market_assessment(self, all_analysis: Dict) -> str:
        """Generate overall market assessment from analyzed stocks"""
        # Calculate market breadth
        bullish_stocks = sum(1 for a in all_analysis.values() if a['trend_score'] > 60)
        total_stocks = len(all_analysis)
        
        breadth_pct = bullish_stocks / total_stocks * 100 if total_stocks > 0 else 50
        
        if breadth_pct > 70:
            return f"Strong market breadth ({breadth_pct:.0f}% bullish). Favorable conditions for momentum strategies."
        elif breadth_pct > 50:
            return f"Moderate market breadth ({breadth_pct:.0f}% bullish). Selective stock picking recommended."
        else:
            return f"Weak market breadth ({breadth_pct:.0f}% bullish). Consider defensive positioning."
    
    def get_sector_analysis(self) -> Dict:
        """Analyze performance by sector"""
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Financial': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'CVS', 'ABBV'],
            'Energy': ['XOM', 'CVX', 'COP']
        }
        
        sector_performance = {}
        
        for sector, stocks in sectors.items():
            sector_scores = []
            for symbol in stocks:
                try:
                    analysis = self._analyze_stock(symbol, 30)
                    if analysis:
                        score = self._calculate_stock_score(analysis, 'balanced')
                        sector_scores.append(score)
                except:
                    continue
            
            if sector_scores:
                sector_performance[sector] = {
                    'average_score': np.mean(sector_scores),
                    'best_stock': stocks[np.argmax(sector_scores)] if sector_scores else None,
                    'stock_count': len(sector_scores)
                }
        
        return sector_performance