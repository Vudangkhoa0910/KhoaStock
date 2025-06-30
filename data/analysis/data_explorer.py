import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class StockDataExplorer:
    def __init__(self, collected_data_dir: str = None, processed_data_dir: str = None):
        self.setup_logging()
        
        if collected_data_dir is None or processed_data_dir is None:
            current_dir = Path(__file__).parent.parent
            self.collected_dir = current_dir / "collected_data"
            self.processed_dir = current_dir / "processed_data"
        else:
            self.collected_dir = Path(collected_data_dir)
            self.processed_dir = Path(processed_data_dir)
            
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            daily_files = list(self.collected_dir.glob(f"daily/{symbol}_daily*.csv"))
            if not daily_files:
                return None
            latest_file = max(daily_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            df['time'] = pd.to_datetime(df['time'])
            return df
        except Exception as e:
            logging.error(f"Lỗi khi load dữ liệu ngày của {symbol}: {str(e)}")
            return None

    def load_processed_signals(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            signal_files = list(self.processed_dir.glob(f"swing_signals/{symbol}_swing_signals*.csv"))
            if not signal_files:
                return None
            latest_file = max(signal_files, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest_file)
        except Exception as e:
            logging.error(f"Lỗi khi load tín hiệu của {symbol}: {str(e)}")
            return None

    def analyze_price_distribution(self, symbol: str) -> Dict:
        df = self.load_daily_data(symbol)
        if df is None:
            return {}
            
        price_stats = {
            'mean': df['close'].mean(),
            'median': df['close'].median(),
            'std': df['close'].std(),
            'skew': stats.skew(df['close']),
            'kurtosis': stats.kurtosis(df['close']),
            'min': df['close'].min(),
            'max': df['close'].max()
        }
        
        # Vẽ histogram phân phối giá
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='close', bins=50)
        plt.title(f'Phân phối giá của {symbol}')
        plt.savefig(self.processed_dir / f"analysis/{symbol}_price_dist.png")
        plt.close()
        
        return price_stats

    def analyze_volume_patterns(self, symbol: str) -> Dict:
        df = self.load_daily_data(symbol)
        if df is None:
            return {}
            
        # Thống kê mô tả về volume
        volume_stats = {
            'mean_volume': df['volume'].mean(),
            'median_volume': df['volume'].median(),
            'std_volume': df['volume'].std(),
            'max_volume': df['volume'].max(),
            'min_volume': df['volume'].min()
        }
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['time'], df['volume'])
        plt.title(f'Khối lượng giao dịch theo thời gian của {symbol}')
        plt.xticks(rotation=45)
        plt.savefig(self.processed_dir / f"analysis/{symbol}_volume_trend.png")
        plt.close()
        
        return volume_stats

    def analyze_price_momentum(self, symbol: str) -> Dict:
        signals_df = self.load_processed_signals(symbol)
        if signals_df is None:
            return {}
            
        momentum_stats = {
            'mean_rsi': signals_df['rsi'].mean(),
            'overbought_periods': len(signals_df[signals_df['rsi'] > 70]),
            'oversold_periods': len(signals_df[signals_df['rsi'] < 30]),
            'positive_macd_periods': len(signals_df[signals_df['macd'] > 0]),
            'negative_macd_periods': len(signals_df[signals_df['macd'] < 0])
        }
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(signals_df.index, signals_df['rsi'])
        ax1.axhline(y=70, color='r', linestyle='--')
        ax1.axhline(y=30, color='g', linestyle='--')
        ax1.set_title(f'RSI của {symbol}')
        
        ax2.plot(signals_df.index, signals_df['macd'])
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.set_title(f'MACD của {symbol}')
        
        plt.tight_layout()
        plt.savefig(self.processed_dir / f"analysis/{symbol}_momentum.png")
        plt.close()
        
        return momentum_stats

    def analyze_support_resistance(self, symbol: str) -> Dict:
        df = self.load_daily_data(symbol)
        if df is None:
            return {}
            
        price_levels = pd.qcut(df['close'], q=10)
        level_stats = price_levels.value_counts().sort_index()
        
        support_resistance = {
            'strong_support': level_stats.index[0].left,
            'weak_support': level_stats.index[2].left,
            'weak_resistance': level_stats.index[7].right,
            'strong_resistance': level_stats.index[9].right,
            'most_traded_range': f"{level_stats.index[level_stats.argmax()].left:.2f} - {level_stats.index[level_stats.argmax()].right:.2f}"
        }
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='close', bins=50)
        plt.axvline(x=support_resistance['strong_support'], color='g', linestyle='--', label='Strong Support')
        plt.axvline(x=support_resistance['strong_resistance'], color='r', linestyle='--', label='Strong Resistance')
        plt.title(f'Các mức giá quan trọng của {symbol}')
        plt.legend()
        plt.savefig(self.processed_dir / f"analysis/{symbol}_levels.png")
        plt.close()
        
        return support_resistance

    def analyze_signal_distribution(self, symbol: str) -> Dict:
        signals_df = self.load_processed_signals(symbol)
        if signals_df is None:
            return {}
            
        signal_counts = signals_df['swing_strength'].value_counts()
        signal_stats = {
            'total_signals': len(signals_df),
            'buy_signals': len(signals_df[signals_df['swing_strength'].isin(['Buy', 'Strong Buy'])]),
            'sell_signals': len(signals_df[signals_df['swing_strength'].isin(['Sell', 'Strong Sell'])]),
            'neutral_signals': len(signals_df[signals_df['swing_strength'] == 'Neutral']),
            'signal_distribution': signal_counts.to_dict()
        }
        
        plt.figure(figsize=(10, 6))
        signal_counts.plot(kind='bar')
        plt.title(f'Phân phối tín hiệu giao dịch của {symbol}')
        plt.xticks(rotation=45)
        plt.savefig(self.processed_dir / f"analysis/{symbol}_signals_dist.png")
        plt.close()
        
        return signal_stats

    def analyze_correlation_matrix(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        price_data = {}
        for symbol in symbols:
            df = self.load_daily_data(symbol)
            if df is not None:
                price_data[symbol] = df.set_index('time')['close']
                
        if not price_data:
            return None
            
        price_df = pd.DataFrame(price_data)
        corr_matrix = price_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Ma trận tương quan giữa các mã')
        plt.savefig(self.processed_dir / "analysis/correlation_matrix.png")
        plt.close()
        
        return corr_matrix

    def generate_summary_report(self, symbols: List[str]) -> Dict:
        report = {}
        for symbol in symbols:
            symbol_report = {
                'price_stats': self.analyze_price_distribution(symbol),
                'volume_stats': self.analyze_volume_patterns(symbol),
                'momentum_stats': self.analyze_price_momentum(symbol),
                'support_resistance': self.analyze_support_resistance(symbol),
                'signal_stats': self.analyze_signal_distribution(symbol)
            }
            report[symbol] = symbol_report
            
        corr_matrix = self.analyze_correlation_matrix(symbols)
        if corr_matrix is not None:
            report['correlation'] = corr_matrix.to_dict()
        
        analysis_dir = self.processed_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        current_date = datetime.now().strftime("%Y%m%d")
        report_file = analysis_dir / f"market_analysis_{current_date}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, cls=NumpyEncoder)
        
        return report

def main():
    explorer = StockDataExplorer()
    
    symbols = ['VCB', 'VNM', 'FPT']
    
    analysis_dir = explorer.processed_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    report = explorer.generate_summary_report(symbols)
    
    for symbol in symbols:
        if symbol in report:
            logging.info(f"\nPhân tích cho {symbol}:")
            logging.info(f"Giá trung bình: {report[symbol]['price_stats'].get('mean', 'N/A'):.2f}")
            logging.info(f"Khối lượng trung bình: {report[symbol]['volume_stats'].get('mean_volume', 'N/A'):.0f}")
            logging.info(f"RSI trung bình: {report[symbol]['momentum_stats'].get('mean_rsi', 'N/A'):.2f}")
            logging.info(f"Vùng giao dịch phổ biến: {report[symbol]['support_resistance'].get('most_traded_range', 'N/A')}")
            logging.info("------------------------")

if __name__ == "__main__":
    main() 