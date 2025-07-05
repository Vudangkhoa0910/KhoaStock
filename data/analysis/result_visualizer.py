# Author: Vu Dang Khoa
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
        self._setup_dirs(processed_data_dir, output_dir)
        plt.style.use('seaborn-v0_8')

    def _setup_dirs(self, processed_data_dir, output_dir):
        if not processed_data_dir or not output_dir:
            root = Path(__file__).parent.parent
            self.processed_dir = root / "processed_data"
            self.output_dir = root / "output"
            return
        self.processed_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def get_model_data(self, symbol: str) -> Dict:
        try:
            model_dir = self.processed_dir / "models"
            data = {}
            importance = model_dir / f"{symbol}_feature_importance.csv"
            if importance.exists():
                data['feature_importance'] = pd.read_csv(importance)
            return data
        except Exception as e:
            logging.error(f"Failed to load model data for {symbol}: {str(e)}")
            return {}

    def plot_features(self, symbols: List[str]):
        plt.figure(figsize=(12, 6))
        for i, sym in enumerate(symbols):
            data = self.get_model_data(sym)
            if 'feature_importance' not in data:
                continue
                
            plt.subplot(1, len(symbols), i+1)
            top = data['feature_importance'].head(5)
            sns.barplot(data=top, x='importance', y='feature')
            plt.title(f'Key Features - {sym}')
            plt.xlabel('Score')
            plt.ylabel('Feature')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()

    def plot_performance(self, metrics: Dict):
        df = pd.DataFrame(metrics).T
        df = df[['accuracy', 'precision', 'recall', 'f1']]
        
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar', width=0.8)
        plt.title('Model Metrics')
        plt.xlabel('Symbol')
        plt.ylabel('Score')
        plt.legend(title='Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance.png')
        plt.close()

    def plot_signals(self, signals: Dict):
        df = pd.DataFrame({
            'Symbol': list(signals.keys()),
            'Signal': [s['signal_strength'] for s in signals.values()],
            'Conf': [s['confidence'] for s in signals.values()]
        })

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(name='Signal', x=df['Symbol'], y=df['Signal']),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(name='Conf', x=df['Symbol'], y=df['Conf'], mode='lines+markers'),
            secondary_y=True
        )

        fig.update_layout(
            title='Signal Analysis',
            xaxis_title='Symbol',
            barmode='group'
        )
        fig.update_yaxes(title_text="Signal", secondary_y=False)
        fig.update_yaxes(title_text="Conf", secondary_y=True)
        fig.write_html(str(self.output_dir / 'trading_signals.html'))

    def plot_technicals(self, signals: Dict):
        data = []
        for sym, sig in signals.items():
            tech = sig['technical_analysis']
            data.append({
                'Symbol': sym,
                'RSI': tech['rsi_level'],
                'Trend': 1 if tech['trend_following'] else 0,
                'Momentum': 1 if tech['momentum'] else 0
            })

        df = pd.DataFrame(data)
        fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('RSI', 'Tech Signals'))
        fig.add_trace(
            go.Bar(name='RSI', x=df['Symbol'], y=df['RSI']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Trend', x=df['Symbol'], y=df['Trend']),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Momentum', x=df['Symbol'], y=df['Momentum']),
            row=2, col=1
        )
        fig.update_layout(title='Technical Analysis Overview')
        fig.write_html(str(self.output_dir / 'technical_analysis.html'))

    def plot_fundamentals(self, signals: Dict):
        data = []
        for sym, sig in signals.items():
            fund = sig['fundamental_analysis']
            data.append({
                'Symbol': sym,
                'P/E': fund.get('pe_ratio', np.nan),
                'ROE': fund.get('roe', np.nan),
                'Revenue Growth': fund.get('revenue_growth', np.nan)
            })

        df = pd.DataFrame(data)
        fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('P/E Ratio', 'ROE', 'Revenue Growth', 'Metrics Comparison'))
        fig.add_trace(
            go.Bar(name='P/E', x=df['Symbol'], y=df['P/E']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='ROE', x=df['Symbol'], y=df['ROE']),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Revenue Growth', x=df['Symbol'], y=df['Revenue Growth']),
            row=2, col=1
        )
        for metric in ['P/E', 'ROE', 'Revenue Growth']:
            fig.add_trace(
                go.Scatter(name=metric, x=df['Symbol'], y=df[metric], mode='lines+markers'),
                row=2, col=2
            )
        fig.update_layout(title='Fundamental Analysis Overview')
        fig.write_html(str(self.output_dir / 'fundamental_analysis.html'))

    def generate_report(self, signals: Dict, metrics: Dict):
        report = {
            'trading_recommendations': {},
            'model_performance': metrics,
            'market_outlook': {
                'overall_sentiment': 0,
                'risk_level': 'Medium',
                'market_trend': 'Sideways'
            }
        }
        for sym, sig in signals.items():
            report['trading_recommendations'][sym] = {
                'action': sig['action'],
                'confidence': sig['confidence'],
                'signal_strength': sig['signal_strength'],
                'key_factors': {
                    'technical': {
                        'trend': 'Uptrend' if sig['technical_analysis']['trend_following'] else 'Downtrend',
                        'rsi': sig['technical_analysis']['rsi_level']
                    },
                    'fundamental': {
                        'revenue_growth': sig['fundamental_analysis'].get('revenue_growth', 'N/A'),
                        'roe': sig['fundamental_analysis'].get('roe', 'N/A')
                    },
                    'sentiment': sig['sentiment_analysis']['interpretation']
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
    visualizer.plot_features(symbols)
    visualizer.plot_performance(performance_metrics)
    visualizer.plot_signals(trading_signals)
    visualizer.plot_technicals(trading_signals)
    visualizer.plot_fundamentals(trading_signals)
    visualizer.generate_report(trading_signals, performance_metrics)
if __name__ == "__main__":
    main()