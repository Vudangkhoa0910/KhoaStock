import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json

class ResultVisualizer:
    def __init__(self, processed_data_dir: str = None, output_dir: str = None):
        if processed_data_dir is None or output_dir is None:
            current_dir = Path(__file__).parent.parent
            self.processed_dir = current_dir / "processed_data"
            self.output_dir = current_dir / "output"
        else:
            self.processed_dir = Path(processed_data_dir)
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8')
    def load_model_results(self, symbol: str) -> Dict:
        try:
            model_dir = self.processed_dir / "models"
            results = {}
            importance_file = model_dir / f"{symbol}_feature_importance.csv"
            if importance_file.exists():
                results['feature_importance'] = pd.read_csv(importance_file)
            return results
        except Exception as e:
            logging.error(f"Lỗi khi load kết quả mô hình cho {symbol}: {str(e)}")
            return {}
    def plot_feature_importance(self, symbols: List[str]):
        plt.figure(figsize=(12, 6))
        for i, symbol in enumerate(symbols):
            results = self.load_model_results(symbol)
            if 'feature_importance' in results:
                importance_df = results['feature_importance']
                top_features = importance_df.head(5)
                plt.subplot(1, len(symbols), i+1)
                sns.barplot(data=top_features, x='importance', y='feature')
                plt.title(f'Top 5 Features - {symbol}')
                plt.xlabel('Importance Score')
                plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
    def plot_model_performance(self, performance_metrics: Dict):
        metrics_df = pd.DataFrame(performance_metrics).T
        metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1']]
        plt.figure(figsize=(10, 6))
        metrics_df.plot(kind='bar', width=0.8)
        plt.title('Model Performance Metrics')
        plt.xlabel('Symbol')
        plt.ylabel('Score')
        plt.legend(title='Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance.png')
        plt.close()
    def plot_trading_signals(self, trading_signals: Dict):
        signals_df = pd.DataFrame({
            'Symbol': list(trading_signals.keys()),
            'Signal Strength': [s['signal_strength'] for s in trading_signals.values()],
            'Confidence': [s['confidence'] for s in trading_signals.values()]
        })
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(name='Signal Strength', x=signals_df['Symbol'], y=signals_df['Signal Strength']),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(name='Confidence', x=signals_df['Symbol'], y=signals_df['Confidence'], mode='lines+markers'),
            secondary_y=True
        )
        fig.update_layout(
            title='Trading Signals Analysis',
            xaxis_title='Symbol',
            barmode='group'
        )
        fig.update_yaxes(title_text="Signal Strength", secondary_y=False)
        fig.update_yaxes(title_text="Confidence", secondary_y=True)
        fig.write_html(str(self.output_dir / 'trading_signals.html'))
    def plot_technical_analysis(self, trading_signals: Dict):
        tech_data = []
        for symbol, signal in trading_signals.items():
            tech = signal['technical_analysis']
            tech_data.append({
                'Symbol': symbol,
                'RSI': tech['rsi_level'],
                'Trend': 1 if tech['trend_following'] else 0,
                'Momentum': 1 if tech['momentum'] else 0
            })
        tech_df = pd.DataFrame(tech_data)
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('RSI Levels', 'Technical Signals'))
        fig.add_trace(
            go.Bar(name='RSI', x=tech_df['Symbol'], y=tech_df['RSI']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Trend', x=tech_df['Symbol'], y=tech_df['Trend']),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Momentum', x=tech_df['Symbol'], y=tech_df['Momentum']),
            row=2, col=1
        )
        fig.update_layout(title='Technical Analysis Overview')
        fig.write_html(str(self.output_dir / 'technical_analysis.html'))
    def plot_fundamental_analysis(self, trading_signals: Dict):
        fund_data = []
        for symbol, signal in trading_signals.items():
            fund = signal['fundamental_analysis']
            fund_data.append({
                'Symbol': symbol,
                'P/E': fund.get('pe_ratio', np.nan),
                'ROE': fund.get('roe', np.nan),
                'Revenue Growth': fund.get('revenue_growth', np.nan)
            })
        fund_df = pd.DataFrame(fund_data)
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('P/E Ratio', 'ROE', 'Revenue Growth', 'Metrics Comparison'))
        fig.add_trace(
            go.Bar(name='P/E', x=fund_df['Symbol'], y=fund_df['P/E']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='ROE', x=fund_df['Symbol'], y=fund_df['ROE']),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Revenue Growth', x=fund_df['Symbol'], y=fund_df['Revenue Growth']),
            row=2, col=1
        )
        for metric in ['P/E', 'ROE', 'Revenue Growth']:
            fig.add_trace(
                go.Scatter(name=metric, x=fund_df['Symbol'], y=fund_df[metric], mode='lines+markers'),
                row=2, col=2
            )
        fig.update_layout(title='Fundamental Analysis Overview')
        fig.write_html(str(self.output_dir / 'fundamental_analysis.html'))
    def generate_summary_report(self, trading_signals: Dict, performance_metrics: Dict):
        report = {
            'trading_recommendations': {},
            'model_performance': performance_metrics,
            'market_outlook': {
                'overall_sentiment': 0,
                'risk_level': 'Medium',
                'market_trend': 'Sideways'
            }
        }
        for symbol, signal in trading_signals.items():
            report['trading_recommendations'][symbol] = {
                'action': signal['action'],
                'confidence': signal['confidence'],
                'signal_strength': signal['signal_strength'],
                'key_factors': {
                    'technical': {
                        'trend': 'Uptrend' if signal['technical_analysis']['trend_following'] else 'Downtrend',
                        'rsi': signal['technical_analysis']['rsi_level']
                    },
                    'fundamental': {
                        'revenue_growth': signal['fundamental_analysis'].get('revenue_growth', 'N/A'),
                        'roe': signal['fundamental_analysis'].get('roe', 'N/A')
                    },
                    'sentiment': signal['sentiment_analysis']['interpretation']
                }
            }
        with open(self.output_dir / 'summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
def main():
    visualizer = ResultVisualizer()
    symbols = ['VCB', 'VNM', 'FPT']
    performance_metrics = {
        'VCB': {'accuracy': 0.71, 'precision': 0.50, 'recall': 0.71, 'f1': 0.59},
        'VNM': {'accuracy': 0.63, 'precision': 0.40, 'recall': 0.63, 'f1': 0.49},
        'FPT': {'accuracy': 0.46, 'precision': 0.21, 'recall': 0.46, 'f1': 0.29}
    }
    trading_signals = {
        'VCB': {
            'action': 'Sell',
            'signal_strength': 2,
            'confidence': 0.49,
            'technical_analysis': {
                'trend_following': False,
                'momentum': True,
                'overbought_oversold': 0,
                'rsi_level': 44.00
            },
            'fundamental_analysis': {
                'pe_ratio': None,
                'roe': None,
                'revenue_growth': 0.0114
            },
            'sentiment_analysis': {
                'score': 0.0,
                'interpretation': 'Negative'
            }
        },
        'VNM': {
            'action': 'Strong Sell',
            'signal_strength': 0,
            'confidence': 0.43,
            'technical_analysis': {
                'trend_following': False,
                'momentum': False,
                'overbought_oversold': 0,
                'rsi_level': 28.89
            },
            'fundamental_analysis': {
                'pe_ratio': None,
                'roe': None,
                'revenue_growth': -0.0296
            },
            'sentiment_analysis': {
                'score': 0.0,
                'interpretation': 'Negative'
            }
        },
        'FPT': {
            'action': 'Sell',
            'signal_strength': 2,
            'confidence': 0.51,
            'technical_analysis': {
                'trend_following': False,
                'momentum': False,
                'overbought_oversold': 0,
                'rsi_level': 42.74
            },
            'fundamental_analysis': {
                'pe_ratio': None,
                'roe': None,
                'revenue_growth': -0.0071
            },
            'sentiment_analysis': {
                'score': 0.0,
                'interpretation': 'Negative'
            }
        }
    }
    visualizer.plot_feature_importance(symbols)
    visualizer.plot_model_performance(performance_metrics)
    visualizer.plot_trading_signals(trading_signals)
    visualizer.plot_technical_analysis(trading_signals)
    visualizer.plot_fundamental_analysis(trading_signals)
    visualizer.generate_summary_report(trading_signals, performance_metrics)
if __name__ == "__main__":
    main()