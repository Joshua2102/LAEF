"""
LAEF Control Interface
Command-line interface to interact with LAEF's daily monitoring and learning system
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import argparse

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from daily_market_monitor import DailyMarketMonitor
    from pattern_recognition_analyzer import PatternRecognitionAnalyzer  
    from knowledge_synthesis import KnowledgeSynthesisEngine
    from prediction_tracker import PredictionTracker
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available.")

class LAEFControlInterface:
    """
    Command-line interface for LAEF monitoring system
    """
    
    def __init__(self):
        self.monitor = None
        self.pattern_analyzer = None
        self.knowledge_engine = None
        self.prediction_tracker = None
        
        # Default paths
        self.knowledge_db_path = "logs/knowledge/market_observations.db"
        self.predictions_db_path = "logs/training/predictions.db"
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LAEF components"""
        try:
            self.monitor = DailyMarketMonitor()
            self.pattern_analyzer = PatternRecognitionAnalyzer(self.knowledge_db_path)
            self.knowledge_engine = KnowledgeSynthesisEngine(self.knowledge_db_path)
            self.prediction_tracker = PredictionTracker(self.predictions_db_path)
            
            print("✅ LAEF components initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            print("Some features may not be available.")
    
    def show_status(self):
        """Show current LAEF system status"""
        print("\n" + "="*60)
        print("🤖 LAEF SYSTEM STATUS")
        print("="*60)
        
        if self.monitor:
            status = self.monitor.get_monitoring_status()
            
            print(f"Monitoring Active: {'🟢 YES' if status['monitoring_active'] else '🔴 NO'}")
            print(f"Daily Scheduler: {'🟢 RUNNING' if status['scheduler_running'] else '🔴 STOPPED'}")
            print(f"Watched Symbols: {len(status['watch_symbols'])}")
            print(f"Next Wake-up: {status['next_wakeup']}")
            
            # Session stats if monitoring is active
            if status['monitoring_active']:
                session_stats = status['current_session_stats']
                print(f"\nCurrent Session:")
                print(f"  Predictions Made: {session_stats['predictions_made']}")
                print(f"  Patterns Detected: {sum(session_stats['patterns_detected'].values())}")
                
                # Show pattern breakdown
                if session_stats['patterns_detected']:
                    print(f"  Pattern Breakdown:")
                    for pattern, count in session_stats['patterns_detected'].items():
                        print(f"    - {pattern}: {count}")
            
            # Knowledge database stats
            knowledge_stats = status.get('knowledge_db_entries', {})
            if knowledge_stats:
                print(f"\nKnowledge Database:")
                for table, count in knowledge_stats.items():
                    print(f"  {table}: {count} entries")
        
        print("="*60)
    
    def show_recent_insights(self, days: int = 7, category: Optional[str] = None):
        """Show recent insights learned by LAEF"""
        print(f"\n🧠 RECENT INSIGHTS (Last {days} days)")
        print("-"*60)
        
        if not self.knowledge_engine:
            print("❌ Knowledge engine not available")
            return
        
        try:
            insights = self.knowledge_engine.get_current_insights(category)
            
            if not insights:
                print("No insights found.")
                return
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_insights = []
            
            for insight in insights:
                try:
                    insight_date = datetime.fromisoformat(insight['timestamp'].replace('Z', ''))
                    if insight_date >= cutoff_date:
                        recent_insights.append(insight)
                except:
                    continue
            
            if not recent_insights:
                print("No recent insights found.")
                return
            
            # Display insights
            for i, insight in enumerate(recent_insights[:10], 1):  # Top 10
                print(f"\n{i}. {insight['content']}")
                print(f"   Category: {insight['insight_type']}")
                print(f"   Confidence: {insight['confidence']:.1%}")
                if insight.get('effectiveness_score'):
                    print(f"   Effectiveness: {insight['effectiveness_score']:.1%}")
                print(f"   Date: {insight['timestamp'][:19]}")
                
                if insight.get('supporting_evidence'):
                    print(f"   Evidence: {', '.join(insight['supporting_evidence'][:2])}")
        
        except Exception as e:
            print(f"❌ Error retrieving insights: {e}")
    
    def show_prediction_performance(self, days: int = 7):
        """Show recent prediction performance"""
        print(f"\n📊 PREDICTION PERFORMANCE (Last {days} days)")
        print("-"*60)
        
        if not self.prediction_tracker:
            print("❌ Prediction tracker not available")
            return
        
        try:
            stats = self.prediction_tracker.get_performance_stats(days=days)
            
            print(f"Total Predictions: {stats['total_predictions']}")
            print(f"Overall Accuracy: {stats['accuracy_rate']:.1%}")
            print(f"Correct: {stats['correct']}")
            print(f"Partially Correct: {stats['partial']}")
            print(f"Incorrect: {stats['incorrect']}")
            print(f"Average Confidence: {stats['avg_confidence']:.3f}")
            print(f"Average Price Error: {stats['avg_price_error']:.2%}")
            
            # Show by symbol if we have data
            print(f"\nTop Performing Symbols:")
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
            
            for symbol in symbols:
                symbol_stats = self.prediction_tracker.get_performance_stats(symbol, days)
                if symbol_stats['total_predictions'] > 0:
                    print(f"  {symbol}: {symbol_stats['accuracy_rate']:.1%} "
                          f"({symbol_stats['total_predictions']} predictions)")
        
        except Exception as e:
            print(f"❌ Error retrieving prediction performance: {e}")
    
    def show_pattern_analysis(self, days: int = 7):
        """Show recent pattern analysis"""
        print(f"\n📈 PATTERN ANALYSIS (Last {days} days)")
        print("-"*60)
        
        if not self.pattern_analyzer:
            print("❌ Pattern analyzer not available")
            return
        
        try:
            insights = self.pattern_analyzer.get_pattern_insights(days)
            
            if not insights:
                print("No pattern data available.")
                return
            
            print(f"Total Patterns Detected: {insights['total_patterns']}")
            print(f"Overall Success Rate: {insights['overall_success_rate']:.1%}")
            
            # Show pattern breakdown
            patterns = insights.get('patterns', [])
            if patterns:
                print(f"\nPattern Performance:")
                for pattern in patterns[:10]:  # Top 10
                    print(f"  {pattern['type']}: {pattern['success_rate']:.1%} success "
                          f"({pattern['total_detected']} detected)")
            
            # Show timeframe distribution
            timeframes = insights.get('timeframe_distribution', {})
            if timeframes:
                print(f"\nTimeframe Distribution:")
                for timeframe, count in timeframes.items():
                    print(f"  {timeframe}: {count} patterns")
        
        except Exception as e:
            print(f"❌ Error retrieving pattern analysis: {e}")
    
    def start_monitoring(self):
        """Start LAEF monitoring manually"""
        print("\n🚀 Starting LAEF monitoring...")
        
        if not self.monitor:
            print("❌ Monitor not available")
            return
        
        try:
            self.monitor.manual_start()
            print("✅ Monitoring started successfully")
            print("💡 Tip: Use 'python laef_control.py status' to check monitoring status")
            
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop LAEF monitoring"""
        print("\n🛑 Stopping LAEF monitoring...")
        
        if not self.monitor:
            print("❌ Monitor not available")
            return
        
        try:
            self.monitor.stop_monitoring()
            print("✅ Monitoring stopped successfully")
            
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")
    
    def start_daily_scheduler(self):
        """Start the daily scheduler"""
        print("\n⏰ Starting daily scheduler...")
        
        if not self.monitor:
            print("❌ Monitor not available")
            return
        
        try:
            self.monitor.start_daily_scheduler()
            print("✅ Daily scheduler started")
            print("📅 LAEF will wake up at 8:00 AM EST every day")
            print("💡 Keep this process running for automatic daily monitoring")
            
            # Keep running
            try:
                while True:
                    time.sleep(300)  # Check every 5 minutes
                    if self.monitor.get_monitoring_status()['monitoring_active']:
                        print(f"📊 Monitoring active... (Press Ctrl+C to stop)")
            except KeyboardInterrupt:
                print("\n🛑 Stopping scheduler...")
                self.monitor.stop_monitoring()
                
        except Exception as e:
            print(f"❌ Error starting daily scheduler: {e}")
    
    def run_knowledge_synthesis(self):
        """Run knowledge synthesis manually"""
        print("\n🧠 Running knowledge synthesis...")
        
        if not self.knowledge_engine:
            print("❌ Knowledge engine not available")
            return
        
        try:
            results = self.knowledge_engine.daily_knowledge_synthesis()
            
            if 'error' in results:
                print(f"❌ Synthesis error: {results['error']}")
                return
            
            synthesis_results = results.get('synthesis_results', {})
            new_insights = synthesis_results.get('new_insights', [])
            
            print(f"✅ Knowledge synthesis complete!")
            print(f"📝 New insights generated: {len(new_insights)}")
            
            if new_insights:
                print(f"\nTop New Insights:")
                for i, insight in enumerate(new_insights[:3], 1):
                    if isinstance(insight, dict):
                        content = insight.get('content', 'Unknown')
                        confidence = insight.get('confidence', 0)
                    else:
                        content = insight.content
                        confidence = insight.confidence
                    print(f"  {i}. {content} (conf: {confidence:.1%})")
            
            print(f"\nTotal insights in knowledge base: {results.get('total_insights', 0)}")
            
        except Exception as e:
            print(f"❌ Error running knowledge synthesis: {e}")
    
    def export_knowledge(self, output_file: str = None):
        """Export LAEF's accumulated knowledge"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"laef_knowledge_export_{timestamp}.json"
        
        print(f"\n💾 Exporting LAEF knowledge to {output_file}...")
        
        try:
            knowledge_export = {
                'export_timestamp': datetime.now().isoformat(),
                'insights': [],
                'pattern_performance': {},
                'prediction_stats': {},
                'market_observations': []
            }
            
            # Export insights
            if self.knowledge_engine:
                insights = self.knowledge_engine.get_current_insights()
                knowledge_export['insights'] = insights
            
            # Export prediction stats
            if self.prediction_tracker:
                stats = self.prediction_tracker.get_performance_stats(days=30)
                knowledge_export['prediction_stats'] = stats
            
            # Export pattern analysis
            if self.pattern_analyzer:
                pattern_insights = self.pattern_analyzer.get_pattern_insights(days=30)
                knowledge_export['pattern_performance'] = pattern_insights
            
            # Export recent market observations
            if os.path.exists(self.knowledge_db_path):
                conn = sqlite3.connect(self.knowledge_db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT observation_type, symbol, data, confidence, timestamp
                    FROM market_observations
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''')
                
                observations = []
                for row in cursor.fetchall():
                    observations.append({
                        'observation_type': row[0],
                        'symbol': row[1],
                        'data': row[2],
                        'confidence': row[3],
                        'timestamp': row[4]
                    })
                
                knowledge_export['market_observations'] = observations
                conn.close()
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(knowledge_export, f, indent=2, default=str)
            
            print(f"✅ Knowledge exported successfully!")
            print(f"📊 Exported {len(knowledge_export['insights'])} insights")
            print(f"📈 Exported {len(knowledge_export['market_observations'])} observations")
            
        except Exception as e:
            print(f"❌ Error exporting knowledge: {e}")
    
    def show_help(self):
        """Show available commands"""
        print("\n" + "="*60)
        print("🤖 LAEF CONTROL INTERFACE - HELP")
        print("="*60)
        print("\nAvailable Commands:")
        print("  status              - Show system status")
        print("  insights [days]     - Show recent insights (default: 7 days)")
        print("  patterns [days]     - Show pattern analysis (default: 7 days)")
        print("  predictions [days]  - Show prediction performance (default: 7 days)")
        print("  start              - Start monitoring manually")
        print("  stop               - Stop monitoring")
        print("  scheduler          - Start daily scheduler (keeps running)")
        print("  synthesize         - Run knowledge synthesis")
        print("  export [filename]  - Export knowledge to JSON")
        print("  help               - Show this help message")
        print("\nExamples:")
        print("  python laef_control.py status")
        print("  python laef_control.py insights 14")
        print("  python laef_control.py export my_knowledge.json")
        print("="*60)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='LAEF Control Interface')
    parser.add_argument('command', nargs='?', help='Command to execute')
    parser.add_argument('args', nargs='*', help='Additional arguments')
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = LAEFControlInterface()
    
    if not args.command:
        print("\n🤖 LAEF Control Interface")
        print("Use 'help' to see available commands")
        interface.show_help()
        return
    
    command = args.command.lower()
    
    try:
        if command == 'status':
            interface.show_status()
            
        elif command == 'insights':
            days = int(args.args[0]) if args.args else 7
            interface.show_recent_insights(days)
            
        elif command == 'patterns':
            days = int(args.args[0]) if args.args else 7
            interface.show_pattern_analysis(days)
            
        elif command == 'predictions':
            days = int(args.args[0]) if args.args else 7
            interface.show_prediction_performance(days)
            
        elif command == 'start':
            interface.start_monitoring()
            
        elif command == 'stop':
            interface.stop_monitoring()
            
        elif command == 'scheduler':
            interface.start_daily_scheduler()
            
        elif command == 'synthesize':
            interface.run_knowledge_synthesis()
            
        elif command == 'export':
            filename = args.args[0] if args.args else None
            interface.export_knowledge(filename)
            
        elif command == 'help':
            interface.show_help()
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'help' to see available commands")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error executing command: {e}")

if __name__ == "__main__":
    main()
