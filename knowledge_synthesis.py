"""
Knowledge Synthesis Engine
Processes daily market observations to extract learnings and improve LAEF's decision making
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class MarketInsight:
    """Structure for storing market insights"""
    insight_type: str
    content: str
    confidence: float
    supporting_evidence: List[str]
    timestamp: datetime
    effectiveness_score: Optional[float] = None

class KnowledgeSynthesisEngine:
    """
    Processes accumulated market observations to extract actionable insights
    and improve LAEF's pattern recognition and decision making
    """
    
    def __init__(self, knowledge_db_path: str):
        self.knowledge_db_path = knowledge_db_path
        
        # Learning parameters
        self.min_observations_for_insight = 10
        self.confidence_threshold = 0.7
        self.pattern_success_threshold = 0.65
        
        # Knowledge categories
        self.insight_categories = [
            'pattern_effectiveness',
            'market_regime_behavior',
            'timing_optimization', 
            'volatility_patterns',
            'sector_relationships',
            'prediction_accuracy_factors'
        ]
        
        # Learned insights storage
        self.insights_cache = {}
        self.load_insights_cache()
        
        # Performance metrics
        self.learning_metrics = {
            'total_insights_generated': 0,
            'insights_by_category': defaultdict(int),
            'prediction_improvement_rate': 0.0,
            'pattern_recognition_accuracy': 0.0
        }
        
    def daily_knowledge_synthesis(self) -> Dict[str, Any]:
        """
        Perform daily knowledge synthesis from accumulated observations
        """
        logger.info("🧠 Starting daily knowledge synthesis...")
        
        synthesis_results = {
            'new_insights': [],
            'updated_insights': [],
            'pattern_learnings': {},
            'prediction_improvements': {},
            'market_regime_updates': {}
        }
        
        try:
            # 1. Analyze pattern effectiveness
            pattern_insights = self._analyze_pattern_effectiveness()
            synthesis_results['pattern_learnings'] = pattern_insights
            
            # 2. Market regime behavior analysis
            regime_insights = self._analyze_market_regime_behavior()
            synthesis_results['market_regime_updates'] = regime_insights
            
            # 3. Prediction accuracy analysis
            prediction_insights = self._analyze_prediction_accuracy()
            synthesis_results['prediction_improvements'] = prediction_insights
            
            # 4. Timing optimization insights
            timing_insights = self._analyze_timing_effectiveness()
            
            # 5. Generate new actionable insights
            new_insights = self._generate_actionable_insights(
                pattern_insights, regime_insights, prediction_insights, timing_insights
            )
            synthesis_results['new_insights'] = new_insights
            
            # 6. Update existing insights
            updated_insights = self._update_existing_insights()
            synthesis_results['updated_insights'] = updated_insights
            
            # 7. Save insights to cache
            self._update_insights_cache(new_insights + updated_insights)
            
            # 8. Generate learning report
            learning_report = self._generate_learning_report(synthesis_results)
            
            logger.info(f"✅ Knowledge synthesis complete: {len(new_insights)} new insights, "
                       f"{len(updated_insights)} updated insights")
            
            return {
                'synthesis_results': synthesis_results,
                'learning_report': learning_report,
                'total_insights': len(self.insights_cache),
                'categories_learned': list(self.learning_metrics['insights_by_category'].keys())
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge synthesis: {e}")
            return {'error': str(e)}
    
    def _analyze_pattern_effectiveness(self) -> Dict[str, Any]:
        """Analyze which patterns are most effective under what conditions"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            
            # Get pattern outcomes from last 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            query = '''
                SELECT pattern_type, pattern_data, outcome, accuracy_score, timestamp
                FROM pattern_observations 
                WHERE timestamp > ? AND outcome != 'pending'
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(thirty_days_ago,))
            conn.close()
            
            if len(df) < self.min_observations_for_insight:
                return {'status': 'insufficient_data', 'patterns_analyzed': 0}
            
            pattern_analysis = {}
            
            # Analyze each pattern type
            for pattern_type in df['pattern_type'].unique():
                pattern_data = df[df['pattern_type'] == pattern_type]
                
                success_rate = (pattern_data['outcome'] == 'success').mean()
                avg_accuracy = pattern_data['accuracy_score'].mean()
                total_occurrences = len(pattern_data)
                
                # Analyze context conditions for successful patterns
                successful_patterns = pattern_data[pattern_data['outcome'] == 'success']
                context_conditions = self._extract_context_conditions(successful_patterns)
                
                pattern_analysis[pattern_type] = {
                    'success_rate': success_rate,
                    'avg_accuracy_score': avg_accuracy,
                    'total_occurrences': total_occurrences,
                    'optimal_conditions': context_conditions,
                    'effectiveness_rating': self._calculate_effectiveness_rating(
                        success_rate, avg_accuracy, total_occurrences
                    )
                }
            
            # Identify best and worst performing patterns
            best_patterns = sorted(
                pattern_analysis.items(), 
                key=lambda x: x[1]['effectiveness_rating'], 
                reverse=True
            )[:3]
            
            worst_patterns = sorted(
                pattern_analysis.items(),
                key=lambda x: x[1]['effectiveness_rating']
            )[:3]
            
            return {
                'status': 'success',
                'patterns_analyzed': len(pattern_analysis),
                'pattern_effectiveness': pattern_analysis,
                'best_patterns': best_patterns,
                'worst_patterns': worst_patterns,
                'overall_pattern_success_rate': df[df['outcome'] == 'success'].shape[0] / len(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pattern effectiveness: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_market_regime_behavior(self) -> Dict[str, Any]:
        """Analyze how different strategies perform in different market regimes"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            
            # Get market regime observations
            query = '''
                SELECT data, timestamp
                FROM market_observations 
                WHERE observation_type = 'regime_change'
                ORDER BY timestamp DESC
                LIMIT 50
            '''
            
            cursor = conn.cursor()
            cursor.execute(query)
            regime_data = cursor.fetchall()
            
            conn.close()
            
            if len(regime_data) < 5:
                return {'status': 'insufficient_regime_data'}
            
            regime_analysis = {
                'regime_transitions': [],
                'regime_performance': {},
                'optimal_strategies_by_regime': {}
            }
            
            # Process regime changes
            for data_json, timestamp in regime_data:
                try:
                    regime_info = json.loads(data_json)
                    regime_analysis['regime_transitions'].append({
                        'timestamp': timestamp,
                        'from_regime': regime_info.get('from_regime'),
                        'to_regime': regime_info.get('to_regime'),
                        'volatility': regime_info.get('volatility'),
                        'trend': regime_info.get('trend')
                    })
                except:
                    continue
            
            # Analyze prediction performance by regime
            regime_performance = self._correlate_predictions_with_regimes()
            regime_analysis['regime_performance'] = regime_performance
            
            return {
                'status': 'success',
                'regime_analysis': regime_analysis,
                'total_regime_changes': len(regime_analysis['regime_transitions'])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime behavior: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_prediction_accuracy(self) -> Dict[str, Any]:
        """Analyze prediction accuracy patterns to identify improvement areas"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            
            # Get prediction data from tracker
            query = '''
                SELECT symbol, timeframe, prediction_accuracy, 
                       confidence, actual_return, predicted_price, current_price,
                       timestamp, technical_indicators
                FROM predictions 
                WHERE prediction_accuracy != 'pending' 
                AND timestamp > ?
                ORDER BY timestamp DESC
            '''
            
            seven_days_ago = datetime.now() - timedelta(days=7)
            df = pd.read_sql_query(query, conn, params=(seven_days_ago,))
            conn.close()
            
            if len(df) < 20:
                return {'status': 'insufficient_prediction_data'}
            
            accuracy_analysis = {
                'overall_accuracy': (df['prediction_accuracy'] == 'correct').mean(),
                'accuracy_by_timeframe': {},
                'accuracy_by_symbol': {},
                'confidence_correlation': {},
                'improvement_opportunities': []
            }
            
            # Accuracy by timeframe
            for timeframe in df['timeframe'].unique():
                timeframe_data = df[df['timeframe'] == timeframe]
                accuracy_analysis['accuracy_by_timeframe'][timeframe] = {
                    'accuracy': (timeframe_data['prediction_accuracy'] == 'correct').mean(),
                    'total_predictions': len(timeframe_data),
                    'avg_confidence': timeframe_data['confidence'].mean()
                }
            
            # Accuracy by symbol
            for symbol in df['symbol'].unique():
                if len(df[df['symbol'] == symbol]) >= 5:  # Minimum predictions
                    symbol_data = df[df['symbol'] == symbol]
                    accuracy_analysis['accuracy_by_symbol'][symbol] = {
                        'accuracy': (symbol_data['prediction_accuracy'] == 'correct').mean(),
                        'total_predictions': len(symbol_data),
                        'avg_confidence': symbol_data['confidence'].mean()
                    }
            
            # Confidence correlation with accuracy
            correct_predictions = df[df['prediction_accuracy'] == 'correct']
            incorrect_predictions = df[df['prediction_accuracy'] == 'incorrect']
            
            if len(correct_predictions) > 0 and len(incorrect_predictions) > 0:
                accuracy_analysis['confidence_correlation'] = {
                    'avg_confidence_correct': correct_predictions['confidence'].mean(),
                    'avg_confidence_incorrect': incorrect_predictions['confidence'].mean(),
                    'confidence_predictive_value': (
                        correct_predictions['confidence'].mean() > 
                        incorrect_predictions['confidence'].mean()
                    )
                }
            
            # Identify improvement opportunities
            improvement_opportunities = []
            
            # Low accuracy timeframes
            for timeframe, data in accuracy_analysis['accuracy_by_timeframe'].items():
                if data['accuracy'] < 0.5 and data['total_predictions'] >= 5:
                    improvement_opportunities.append(
                        f"Low accuracy in {timeframe} predictions: {data['accuracy']:.1%}"
                    )
            
            # Overconfident predictions
            if (accuracy_analysis.get('confidence_correlation', {}).get('confidence_predictive_value') == False):
                improvement_opportunities.append(
                    "Model showing overconfidence - calibration needed"
                )
            
            accuracy_analysis['improvement_opportunities'] = improvement_opportunities
            
            return {
                'status': 'success',
                'accuracy_analysis': accuracy_analysis,
                'total_predictions_analyzed': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing prediction accuracy: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_timing_effectiveness(self) -> Dict[str, Any]:
        """Analyze optimal timing for different types of predictions/trades"""
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            
            # Get observations with timing data
            query = '''
                SELECT observation_type, data, timestamp, confidence
                FROM market_observations
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            '''
            
            week_ago = datetime.now() - timedelta(days=7)
            cursor = conn.cursor()
            cursor.execute(query, (week_ago,))
            observations = cursor.fetchall()
            conn.close()
            
            timing_analysis = {
                'best_times_for_predictions': {},
                'market_session_effectiveness': {},
                'day_of_week_patterns': {},
                'volatility_timing': {}
            }
            
            # Analyze by hour of day
            hourly_success = defaultdict(list)
            
            for obs_type, data_json, timestamp, confidence in observations:
                try:
                    hour = datetime.fromisoformat(timestamp).hour
                    hourly_success[hour].append(confidence)
                except:
                    continue
            
            # Calculate effectiveness by hour
            for hour, confidences in hourly_success.items():
                if len(confidences) >= 3:
                    timing_analysis['best_times_for_predictions'][f"{hour:02d}:00"] = {
                        'avg_confidence': np.mean(confidences),
                        'observations': len(confidences)
                    }
            
            # Market session analysis (simplified)
            session_mapping = {
                range(9, 11): 'market_open',
                range(11, 14): 'mid_day', 
                range(14, 16): 'afternoon',
                range(16, 17): 'after_hours'
            }
            
            session_performance = defaultdict(list)
            for hour, confidences in hourly_success.items():
                for hour_range, session in session_mapping.items():
                    if hour in hour_range:
                        session_performance[session].extend(confidences)
            
            for session, confidences in session_performance.items():
                if confidences:
                    timing_analysis['market_session_effectiveness'][session] = {
                        'avg_effectiveness': np.mean(confidences),
                        'total_observations': len(confidences)
                    }
            
            return {
                'status': 'success',
                'timing_analysis': timing_analysis,
                'total_observations': len(observations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timing effectiveness: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_actionable_insights(self, pattern_insights: Dict, regime_insights: Dict,
                                    prediction_insights: Dict, timing_insights: Dict) -> List[MarketInsight]:
        """Generate actionable insights from analysis results"""
        insights = []
        
        try:
            # Pattern effectiveness insights
            if pattern_insights.get('status') == 'success':
                best_patterns = pattern_insights.get('best_patterns', [])
                worst_patterns = pattern_insights.get('worst_patterns', [])
                
                if best_patterns:
                    best_pattern = best_patterns[0]
                    insights.append(MarketInsight(
                        insight_type='pattern_effectiveness',
                        content=f"Most effective pattern: {best_pattern[0]} with {best_pattern[1]['success_rate']:.1%} success rate",
                        confidence=0.8,
                        supporting_evidence=[f"Based on {best_pattern[1]['total_occurrences']} occurrences"],
                        timestamp=datetime.now()
                    ))
                
                if worst_patterns:
                    worst_pattern = worst_patterns[0]
                    insights.append(MarketInsight(
                        insight_type='pattern_effectiveness',
                        content=f"Least effective pattern: {worst_pattern[0]} with {worst_pattern[1]['success_rate']:.1%} success rate - consider avoiding",
                        confidence=0.7,
                        supporting_evidence=[f"Based on {worst_pattern[1]['total_occurrences']} occurrences"],
                        timestamp=datetime.now()
                    ))
            
            # Prediction accuracy insights
            if prediction_insights.get('status') == 'success':
                accuracy_data = prediction_insights['accuracy_analysis']
                improvement_ops = accuracy_data.get('improvement_opportunities', [])
                
                for opportunity in improvement_ops:
                    insights.append(MarketInsight(
                        insight_type='prediction_accuracy_factors',
                        content=f"Improvement opportunity: {opportunity}",
                        confidence=0.75,
                        supporting_evidence=[f"Analysis of {prediction_insights['total_predictions_analyzed']} predictions"],
                        timestamp=datetime.now()
                    ))
                
                # Best timeframe insight
                timeframe_accuracies = accuracy_data.get('accuracy_by_timeframe', {})
                if timeframe_accuracies:
                    best_timeframe = max(timeframe_accuracies.keys(), 
                                       key=lambda k: timeframe_accuracies[k]['accuracy'])
                    insights.append(MarketInsight(
                        insight_type='timing_optimization',
                        content=f"Most accurate prediction timeframe: {best_timeframe} ({timeframe_accuracies[best_timeframe]['accuracy']:.1%})",
                        confidence=0.8,
                        supporting_evidence=[f"{timeframe_accuracies[best_timeframe]['total_predictions']} predictions analyzed"],
                        timestamp=datetime.now()
                    ))
            
            # Timing insights
            if timing_insights.get('status') == 'success':
                timing_data = timing_insights['timing_analysis']
                session_effectiveness = timing_data.get('market_session_effectiveness', {})
                
                if session_effectiveness:
                    best_session = max(session_effectiveness.keys(),
                                     key=lambda k: session_effectiveness[k]['avg_effectiveness'])
                    insights.append(MarketInsight(
                        insight_type='timing_optimization',
                        content=f"Most effective trading session: {best_session}",
                        confidence=0.7,
                        supporting_evidence=[f"Based on {session_effectiveness[best_session]['total_observations']} observations"],
                        timestamp=datetime.now()
                    ))
            
            # Regime insights
            if regime_insights.get('status') == 'success':
                regime_changes = regime_insights['regime_analysis'].get('total_regime_changes', 0)
                if regime_changes > 0:
                    insights.append(MarketInsight(
                        insight_type='market_regime_behavior',
                        content=f"Market regime instability detected: {regime_changes} regime changes recently",
                        confidence=0.8,
                        supporting_evidence=[f"Tracked {regime_changes} regime transitions"],
                        timestamp=datetime.now()
                    ))
            
        except Exception as e:
            logger.error(f"Error generating actionable insights: {e}")
        
        return insights
    
    def _update_existing_insights(self) -> List[MarketInsight]:
        """Update existing insights based on new evidence"""
        updated_insights = []
        
        try:
            # Review existing insights and update effectiveness scores
            for insight_id, insight in self.insights_cache.items():
                if isinstance(insight, dict):
                    # Convert dict to MarketInsight if needed
                    try:
                        insight_obj = MarketInsight(**insight)
                    except:
                        continue
                else:
                    insight_obj = insight
                
                # Check if insight is still relevant (not too old)
                age_days = (datetime.now() - insight_obj.timestamp).days
                
                if age_days > 30:
                    # Archive old insights
                    continue
                
                # Update effectiveness score if we have new data
                effectiveness_score = self._calculate_insight_effectiveness(insight_obj)
                
                if effectiveness_score != insight_obj.effectiveness_score:
                    insight_obj.effectiveness_score = effectiveness_score
                    updated_insights.append(insight_obj)
                    
        except Exception as e:
            logger.error(f"Error updating existing insights: {e}")
        
        return updated_insights
    
    def _calculate_insight_effectiveness(self, insight: MarketInsight) -> float:
        """Calculate how effective an insight has been"""
        try:
            # Simplified effectiveness calculation
            # In practice, would correlate insight with subsequent performance
            
            base_score = insight.confidence
            
            # Age penalty
            age_days = (datetime.now() - insight.timestamp).days
            age_penalty = min(age_days * 0.01, 0.3)  # Max 30% penalty
            
            effectiveness = base_score - age_penalty
            
            return max(0.0, min(1.0, effectiveness))
            
        except:
            return 0.5
    
    def _update_insights_cache(self, insights: List[MarketInsight]):
        """Update the insights cache with new insights"""
        try:
            for insight in insights:
                # Use timestamp as key for uniqueness
                key = f"{insight.insight_type}_{insight.timestamp.isoformat()}"
                self.insights_cache[key] = insight
                
                # Update metrics
                self.learning_metrics['total_insights_generated'] += 1
                self.learning_metrics['insights_by_category'][insight.insight_type] += 1
            
            # Save to file
            self.save_insights_cache()
            
        except Exception as e:
            logger.error(f"Error updating insights cache: {e}")
    
    def save_insights_cache(self):
        """Save insights cache to disk"""
        try:
            cache_dir = "logs/knowledge"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, "insights_cache.pkl")
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'insights': self.insights_cache,
                    'metrics': self.learning_metrics
                }, f)
                
        except Exception as e:
            logger.error(f"Error saving insights cache: {e}")
    
    def load_insights_cache(self):
        """Load insights cache from disk"""
        try:
            cache_file = "logs/knowledge/insights_cache.pkl"
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.insights_cache = data.get('insights', {})
                    self.learning_metrics.update(data.get('metrics', {}))
                    
                logger.info(f"Loaded {len(self.insights_cache)} insights from cache")
            
        except Exception as e:
            logger.error(f"Error loading insights cache: {e}")
            self.insights_cache = {}
    
    def get_current_insights(self, category: Optional[str] = None) -> List[Dict]:
        """Get current insights, optionally filtered by category"""
        try:
            insights = []
            
            for key, insight in self.insights_cache.items():
                if isinstance(insight, dict):
                    insight_dict = insight
                else:
                    insight_dict = {
                        'insight_type': insight.insight_type,
                        'content': insight.content,
                        'confidence': insight.confidence,
                        'effectiveness_score': insight.effectiveness_score,
                        'timestamp': insight.timestamp.isoformat(),
                        'supporting_evidence': insight.supporting_evidence
                    }
                
                if category is None or insight_dict.get('insight_type') == category:
                    insights.append(insight_dict)
            
            # Sort by effectiveness score and recency
            insights.sort(key=lambda x: (
                x.get('effectiveness_score', 0), 
                x.get('timestamp', '')
            ), reverse=True)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting current insights: {e}")
            return []
    
    def _generate_learning_report(self, synthesis_results: Dict) -> str:
        """Generate a comprehensive learning report"""
        try:
            report_lines = []
            report_lines.append("="*60)
            report_lines.append("LAEF KNOWLEDGE SYNTHESIS REPORT")
            report_lines.append("="*60)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # New insights
            new_insights = synthesis_results.get('new_insights', [])
            report_lines.append(f"NEW INSIGHTS GENERATED: {len(new_insights)}")
            for insight in new_insights[:5]:  # Top 5
                if isinstance(insight, dict):
                    content = insight.get('content', 'Unknown')
                    confidence = insight.get('confidence', 0)
                else:
                    content = insight.content
                    confidence = insight.confidence
                report_lines.append(f"  • {content} (confidence: {confidence:.1%})")
            report_lines.append("")
            
            # Pattern learnings
            pattern_learnings = synthesis_results.get('pattern_learnings', {})
            if pattern_learnings.get('status') == 'success':
                report_lines.append("PATTERN EFFECTIVENESS ANALYSIS:")
                best_patterns = pattern_learnings.get('best_patterns', [])
                if best_patterns:
                    best = best_patterns[0]
                    report_lines.append(f"  Best Pattern: {best[0]} ({best[1]['success_rate']:.1%} success)")
                
                overall_success = pattern_learnings.get('overall_pattern_success_rate', 0)
                report_lines.append(f"  Overall Pattern Success Rate: {overall_success:.1%}")
                report_lines.append("")
            
            # Prediction improvements
            pred_improvements = synthesis_results.get('prediction_improvements', {})
            if pred_improvements.get('status') == 'success':
                accuracy_analysis = pred_improvements['accuracy_analysis']
                report_lines.append("PREDICTION ACCURACY ANALYSIS:")
                report_lines.append(f"  Overall Accuracy: {accuracy_analysis['overall_accuracy']:.1%}")
                
                # Best timeframes
                timeframe_acc = accuracy_analysis.get('accuracy_by_timeframe', {})
                if timeframe_acc:
                    best_tf = max(timeframe_acc.keys(), key=lambda k: timeframe_acc[k]['accuracy'])
                    report_lines.append(f"  Best Timeframe: {best_tf} ({timeframe_acc[best_tf]['accuracy']:.1%})")
                report_lines.append("")
            
            # Learning metrics
            report_lines.append("CUMULATIVE LEARNING METRICS:")
            report_lines.append(f"  Total Insights Generated: {self.learning_metrics['total_insights_generated']}")
            report_lines.append(f"  Active Insight Categories: {len(self.learning_metrics['insights_by_category'])}")
            report_lines.append(f"  Insights in Cache: {len(self.insights_cache)}")
            report_lines.append("")
            
            report_lines.append("="*60)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating learning report: {e}")
            return f"Error generating report: {e}"
    
    def _extract_context_conditions(self, successful_patterns: pd.DataFrame) -> Dict[str, Any]:
        """Extract context conditions from successful patterns"""
        try:
            conditions = {}
            
            # This would analyze the pattern_data JSON to find common conditions
            # Simplified version
            if len(successful_patterns) > 0:
                conditions['sample_size'] = len(successful_patterns)
                conditions['avg_confidence'] = successful_patterns['accuracy_score'].mean()
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error extracting context conditions: {e}")
            return {}
    
    def _calculate_effectiveness_rating(self, success_rate: float, avg_accuracy: float, 
                                      occurrences: int) -> float:
        """Calculate overall effectiveness rating for a pattern"""
        try:
            # Weighted score considering multiple factors
            score = (
                success_rate * 0.4 +  # Success rate is most important
                avg_accuracy * 0.3 +  # Average accuracy score
                min(occurrences / 20, 1.0) * 0.3  # Sample size reliability (max at 20+ occurrences)
            )
            
            return score
            
        except:
            return 0.0
    
    def _correlate_predictions_with_regimes(self) -> Dict[str, Any]:
        """Correlate prediction performance with market regimes"""
        try:
            # This would require joining prediction data with regime data
            # Simplified implementation
            return {
                'correlation_analysis': 'pending_implementation',
                'regime_count': 0
            }
            
        except Exception as e:
            logger.error(f"Error correlating predictions with regimes: {e}")
            return {'error': str(e)}
