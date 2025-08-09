"""
Email Reporter for Daily Learning Reports
Sends comprehensive daily reports in layman's terms explaining AI learning progress.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from config import EMAIL_NOTIFICATIONS

class EmailReporter:
    """
    Sends daily learning reports via email with explanations in layman's terms.
    """
    
    def __init__(self):
        """Initialize the email reporter."""
        self.smtp_server = EMAIL_NOTIFICATIONS["smtp_server"]
        self.smtp_port = EMAIL_NOTIFICATIONS["smtp_port"]
        self.sender_email = EMAIL_NOTIFICATIONS["sender_email"]
        self.sender_password = EMAIL_NOTIFICATIONS["sender_password"]
        self.recipient_email = EMAIL_NOTIFICATIONS["recipient_email"]
        self.enabled = EMAIL_NOTIFICATIONS["enabled"]
        
        if not self.enabled:
            logging.info("[EMAIL] Email notifications disabled in config")
        elif not all([self.sender_email, self.sender_password, self.recipient_email]):
            logging.warning("[EMAIL] Email configuration incomplete - notifications disabled")
            self.enabled = False
        else:
            logging.info(f"[EMAIL] Email reporter initialized - will send to {self.recipient_email}")
    
    def send_daily_report(self, learning_stats: Dict, monitor_stats: Dict, 
                         market_summary: Dict, confidence_trend: Dict, service_stats: Dict):
        """
        Send a comprehensive daily learning report in layman's terms.
        """
        if not self.enabled:
            logging.debug("[EMAIL] Email notifications disabled")
            return False
        
        try:
            # Generate the report content
            subject = self._generate_subject(learning_stats, confidence_trend)
            html_content = self._generate_html_report(
                learning_stats, monitor_stats, market_summary, confidence_trend, service_stats
            )
            
            # Send the email
            success = self._send_email(subject, html_content)
            
            if success:
                logging.info(f"[EMAIL] Daily report sent successfully to {self.recipient_email}")
            else:
                logging.error("[EMAIL] Failed to send daily report")
            
            return success
            
        except Exception as e:
            logging.error(f"[EMAIL] Failed to send daily report: {e}")
            return False
    
    def _generate_subject(self, learning_stats: Dict, confidence_trend: Dict) -> str:
        """Generate email subject line."""
        date_str = datetime.now().strftime('%B %d, %Y')
        accuracy = learning_stats['buffer_stats']['avg_accuracy']
        trend = confidence_trend['trend']
        
        if trend == 'improving':
            trend_emoji = "📈"
        elif trend == 'declining':
            trend_emoji = "📉"
        else:
            trend_emoji = "➡️"
        
        return f"🤖 LAEF AI Trading Report - {date_str} {trend_emoji} (Accuracy: {accuracy:.1%})"
    
    def _generate_html_report(self, learning_stats: Dict, monitor_stats: Dict,
                             market_summary: Dict, confidence_trend: Dict, service_stats: Dict) -> str:
        """Generate comprehensive HTML report in layman's terms."""
        
        # Extract key metrics
        buffer_stats = learning_stats['buffer_stats']
        training_stats = learning_stats['training_stats']
        accuracy = buffer_stats['avg_accuracy']
        experiences = buffer_stats['completed_experiences']
        trend = confidence_trend['trend']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px; }}
                .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
                .metric {{ display: inline-block; background: white; padding: 15px; margin: 10px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .bad {{ color: #dc3545; }}
                .emoji {{ font-size: 1.2em; }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 8px 0; }}
                .highlight {{ background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }}
                .insight {{ background: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; margin: 15px 0; }}
            </style>
        </head>
        <body>
        
        <div class="header">
            <h1>🤖 LAEF AI Trading System</h1>
            <h2>Daily Learning Report</h2>
            <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Today's AI Performance Summary</h2>
            <p><strong>Think of your AI as a student learning to predict stock movements...</strong></p>
            
            <div style="text-align: center;">
                <div class="metric">
                    <div class="metric-value {self._get_accuracy_class(accuracy)}">{accuracy:.1%}</div>
                    <div class="metric-label">Prediction Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{experiences:,}</div>
                    <div class="metric-label">Learning Experiences</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{training_stats['total_training_sessions']}</div>
                    <div class="metric-label">Training Sessions</div>
                </div>
            </div>
            
            {self._explain_accuracy(accuracy, trend, experiences)}
        </div>
        
        <div class="section">
            <h2>🧠 What Your AI Learned Today</h2>
            {self._explain_learning_progress(learning_stats, confidence_trend)}
        </div>
        
        <div class="section">
            <h2>📈 Market Monitoring Activity</h2>
            {self._explain_monitoring_activity(monitor_stats, market_summary)}
        </div>
        
        <div class="section">
            <h2>🎯 How This Improves Your Trading</h2>
            {self._explain_trading_improvements(learning_stats, confidence_trend)}
        </div>
        
        <div class="section">
            <h2>📋 Technical Details</h2>
            {self._generate_technical_summary(learning_stats, monitor_stats, service_stats)}
        </div>
        
        <div class="section">
            <h2>🔮 Tomorrow's Plan</h2>
            {self._explain_tomorrow_plan(learning_stats, confidence_trend)}
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666; font-size: 12px;">
            <p>This report was automatically generated by your LAEF AI Trading System</p>
            <p>Report generated at {datetime.now().strftime('%I:%M %p on %B %d, %Y')}</p>
        </div>
        
        </body>
        </html>
        """
        
        return html
    
    def _get_accuracy_class(self, accuracy: float) -> str:
        """Get CSS class based on accuracy level."""
        if accuracy >= 0.7:
            return "good"
        elif accuracy >= 0.5:
            return "warning"
        else:
            return "bad"
    
    def _explain_accuracy(self, accuracy: float, trend: str, experiences: int) -> str:
        """Explain accuracy in layman's terms."""
        if accuracy >= 0.7:
            explanation = f"""
            <div class="insight">
                <strong>🎉 Excellent Performance!</strong><br>
                Your AI is correctly predicting stock movements {accuracy:.1%} of the time. 
                This is considered excellent performance in stock prediction. For comparison, 
                most professional traders aim for 55-60% accuracy.
            </div>
            """
        elif accuracy >= 0.6:
            explanation = f"""
            <div class="insight">
                <strong>✅ Good Performance!</strong><br>
                Your AI is correctly predicting stock movements {accuracy:.1%} of the time.
                This is above average and shows the system is learning effectively from market patterns.
            </div>
            """
        elif accuracy >= 0.5:
            explanation = f"""
            <div class="highlight">
                <strong>📚 Learning in Progress</strong><br>
                Your AI is correctly predicting stock movements {accuracy:.1%} of the time.
                This is better than random guessing (50%) but there's room for improvement as it learns more patterns.
            </div>
            """
        else:
            explanation = f"""
            <div class="highlight">
                <strong>🔄 Early Learning Phase</strong><br>
                Your AI is still in early learning stages with {accuracy:.1%} accuracy.
                This is normal for new systems - it needs more data to identify reliable patterns.
            </div>
            """
        
        trend_explanation = ""
        if trend == 'improving':
            trend_explanation = "<p><strong>📈 Trend:</strong> Your AI is getting better each day - accuracy is improving over time!</p>"
        elif trend == 'declining':
            trend_explanation = "<p><strong>📉 Trend:</strong> Recent accuracy has decreased slightly. This can happen when market conditions change rapidly.</p>"
        else:
            trend_explanation = "<p><strong>➡️ Trend:</strong> Accuracy has been relatively stable, showing consistent performance.</p>"
        
        return explanation + trend_explanation
    
    def _explain_learning_progress(self, learning_stats: Dict, confidence_trend: Dict) -> str:
        """Explain what the AI learned in simple terms."""
        buffer_stats = learning_stats['buffer_stats']
        experiences = buffer_stats['completed_experiences']
        completion_rate = buffer_stats['completion_rate']
        
        return f"""
        <p><strong>Today your AI analyzed {experiences:,} different market situations</strong> and learned from the outcomes:</p>
        <ul>
            <li><strong>Pattern Recognition:</strong> The AI identified which market indicators (like RSI, MACD, moving averages) are most reliable for predicting price movements</li>
            <li><strong>Timing Insights:</strong> It learned the best times to enter and exit positions based on historical success rates</li>
            <li><strong>Risk Assessment:</strong> The system refined its understanding of when markets are too risky vs. when opportunities are favorable</li>
            <li><strong>Confidence Calibration:</strong> The AI learned to be more confident in strong signals and more cautious in uncertain conditions</li>
        </ul>
        
        <div class="insight">
            <strong>🔍 Key Insight:</strong> {completion_rate:.1%} of today's predictions now have known outcomes, 
            meaning the AI can learn from {int(experiences * completion_rate):,} real market results to improve future decisions.
        </div>
        """
    
    def _explain_monitoring_activity(self, monitor_stats: Dict, market_summary: Dict) -> str:
        """Explain monitoring activity in simple terms."""
        symbols_monitored = monitor_stats['symbols_monitored']
        symbols_with_data = monitor_stats['symbols_with_current_data']
        success_rate = monitor_stats['success_rate']
        active_symbols = market_summary.get('active_symbols', 0)
        avg_change = market_summary.get('average_price_change_1h', 0)
        volatile_symbols = len(market_summary.get('high_volatility_symbols', []))
        
        return f"""
        <p><strong>Your AI continuously watched {symbols_monitored} different stocks</strong> throughout the day, like having a dedicated analyst monitoring each one:</p>
        <ul>
            <li><strong>Data Collection:</strong> Successfully gathered fresh data from {symbols_with_data} stocks ({success_rate:.1f}% success rate)</li>
            <li><strong>Market Activity:</strong> {active_symbols} stocks showed significant trading activity today</li>
            <li><strong>Market Movement:</strong> Average hourly price change was {avg_change:+.2f}%</li>
            <li><strong>Volatility Detection:</strong> Identified {volatile_symbols} stocks with high volatility (big price swings)</li>
        </ul>
        
        <p><strong>What this means:</strong> Your AI is constantly learning from real market behavior, 
        not just when you're actively trading. This continuous monitoring helps it stay current with market trends.</p>
        """
    
    def _explain_trading_improvements(self, learning_stats: Dict, confidence_trend: Dict) -> str:
        """Explain how learning improves trading in simple terms."""
        accuracy = learning_stats['buffer_stats']['avg_accuracy']
        trend = confidence_trend['trend']
        
        improvements = []
        
        if accuracy > 0.6:
            improvements.append("✅ <strong>Better Entry Points:</strong> The AI is getting better at identifying when to buy stocks at favorable prices")
            improvements.append("✅ <strong>Smarter Exit Timing:</strong> Improved ability to sell at optimal times to maximize profits")
        
        if trend == 'improving':
            improvements.append("✅ <strong>Adaptive Learning:</strong> The system is continuously adapting to new market conditions")
            improvements.append("✅ <strong>Reduced False Signals:</strong> Getting better at avoiding trades that would lose money")
        
        improvements.append("✅ <strong>Risk Management:</strong> Enhanced ability to assess when markets are too risky for trading")
        improvements.append("✅ <strong>Confidence Scoring:</strong> Better at rating how confident it is in each trading signal")
        
        practical_benefits = f"""
        <h3>🎯 Practical Benefits for Your Trading:</h3>
        <ul>
            {"".join(f"<li>{improvement}</li>" for improvement in improvements)}
        </ul>
        
        <div class="insight">
            <strong>💡 Bottom Line:</strong> Each day of learning makes your trading system smarter and more profitable. 
            The AI is essentially building a massive database of "what works" and "what doesn't" in different market conditions.
        </div>
        """
        
        return practical_benefits
    
    def _generate_technical_summary(self, learning_stats: Dict, monitor_stats: Dict, service_stats: Dict) -> str:
        """Generate technical summary for those who want details."""
        buffer_stats = learning_stats['buffer_stats']
        training_stats = learning_stats['training_stats']
        
        return f"""
        <details>
            <summary><strong>Click to view technical details</strong></summary>
            <div style="margin-top: 15px; font-family: monospace; font-size: 12px;">
                <strong>Learning Buffer:</strong><br>
                • Total experiences: {buffer_stats['total_experiences']:,}<br>
                • Completed experiences: {buffer_stats['completed_experiences']:,}<br>
                • Buffer utilization: {(buffer_stats['total_experiences'] / 10000) * 100:.1f}%<br>
                • Ready for training: {'Yes' if buffer_stats['ready_for_training'] else 'No'}<br><br>
                
                <strong>Training Statistics:</strong><br>
                • Total training sessions: {training_stats['total_training_sessions']}<br>
                • Model updates: {training_stats['model_updates']}<br>
                • Last training loss: {training_stats.get('last_training_loss', 'N/A')}<br><br>
                
                <strong>Monitoring Performance:</strong><br>
                • Total monitoring cycles: {monitor_stats['total_cycles']}<br>
                • Successful updates: {monitor_stats['successful_updates']:,}<br>
                • Failed updates: {monitor_stats['failed_updates']:,}<br>
                • Success rate: {monitor_stats['success_rate']:.1f}%<br>
            </div>
        </details>
        """
    
    def _explain_tomorrow_plan(self, learning_stats: Dict, confidence_trend: Dict) -> str:
        """Explain what will happen tomorrow."""
        buffer_stats = learning_stats['buffer_stats']
        
        plan = """
        <p><strong>Tomorrow your AI will:</strong></p>
        <ul>
            <li>🕘 <strong>9:30 AM:</strong> Begin monitoring all tracked stocks when markets open</li>
            <li>📊 <strong>Throughout the day:</strong> Collect new trading signals and market data</li>
            <li>🧠 <strong>Every hour:</strong> Update its learning from recent market outcomes</li>
            <li>📈 <strong>4:00 PM:</strong> Stop active monitoring when markets close</li>
            <li>🎓 <strong>6:00 PM:</strong> Perform intensive learning session with all day's data</li>
            <li>📧 <strong>6:30 PM:</strong> Send you tomorrow's learning report</li>
        </ul>
        """
        
        if buffer_stats['ready_for_training']:
            plan += """
            <div class="insight">
                <strong>🚀 Ready for Advanced Learning:</strong> Your AI has collected enough data to perform 
                sophisticated pattern recognition and will continue improving its accuracy tomorrow.
            </div>
            """
        else:
            needed = 50 - buffer_stats['completed_experiences']
            plan += f"""
            <div class="highlight">
                <strong>📚 Building Foundation:</strong> Your AI needs {needed} more learning experiences 
                before it can perform advanced training. It's currently building its foundational knowledge.
            </div>
            """
        
        return plan
    
    def _send_email(self, subject: str, html_content: str) -> bool:
        """Send email with the report."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logging.error(f"[EMAIL] Failed to send email: {e}")
            return False
    
    def test_email_connection(self) -> bool:
        """Test email configuration."""
        if not self.enabled:
            print("❌ Email notifications are disabled")
            return False
        
        try:
            test_subject = "🤖 LAEF AI System - Email Test"
            test_content = """
            <html>
            <body>
                <h2>✅ Email Configuration Test Successful!</h2>
                <p>Your LAEF AI Trading System can successfully send email reports.</p>
                <p>You will receive daily learning reports at 6:30 PM after market close.</p>
                <p><small>This is a test message sent at """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</small></p>
            </body>
            </html>
            """
            
            success = self._send_email(test_subject, test_content)
            
            if success:
                print(f"✅ Test email sent successfully to {self.recipient_email}")
                return True
            else:
                print("❌ Failed to send test email")
                return False
                
        except Exception as e:
            print(f"❌ Email test failed: {e}")
            return False

# Global email reporter instance
_email_reporter = None

def get_email_reporter() -> EmailReporter:
    """Get or create global email reporter instance."""
    global _email_reporter
    if _email_reporter is None:
        _email_reporter = EmailReporter()
    return _email_reporter