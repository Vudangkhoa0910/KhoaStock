import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import ta
from scipy import stats
import os

class SwingTradeDataProcessor:
    def __init__(self, collected_data_dir: str = None):
        """
        Khởi tạo processor với các thiết lập cơ bản
        
        Args:
            collected_data_dir: Thư mục chứa dữ liệu đã thu thập
        """
        self.setup_logging()
        
        if collected_data_dir is None:
            current_dir = Path(__file__).parent
            data_dir = current_dir.parent
            self.base_dir = data_dir / "collected_data"
        else:
            self.base_dir = Path(collected_data_dir)
            
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục {self.base_dir}")
            
        logging.info(f"Đường dẫn thu thập dữ liệu: {self.base_dir}")
        
        self.processed_dir = data_dir / "processed_data"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Các thư mục con để lưu dữ liệu đã xử lý
        self.swing_signals_dir = self.processed_dir / "swing_signals"
        self.volume_analysis_dir = self.processed_dir / "volume_analysis"
        self.support_resistance_dir = self.processed_dir / "support_resistance"
        self.momentum_dir = self.processed_dir / "momentum"
        
        for dir_path in [
            self.swing_signals_dir,
            self.volume_analysis_dir,
            self.support_resistance_dir,
            self.momentum_dir
        ]:
            dir_path.mkdir(exist_ok=True)

    def setup_logging(self):
        """Thiết lập logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load và tiền xử lý dữ liệu ngày
        
        Args:
            symbol: Mã chứng khoán
            
        Returns:
            DataFrame đã được tiền xử lý hoặc None nếu có lỗi
        """
        try:
            # Load dữ liệu từ thư mục daily
            daily_files = list(self.base_dir.glob(f"daily/{symbol}_daily*.csv"))
            if not daily_files:
                logging.warning(f"Không tìm thấy dữ liệu ngày cho {symbol}")
                return None
                
            # Lấy file mới nhất
            latest_file = max(daily_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            
            # Chuyển đổi cột thời gian
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # Sắp xếp theo thời gian
            df.sort_index(inplace=True)
            
            # Thêm các cột phụ trợ
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            logging.error(f"Lỗi khi load dữ liệu ngày cho {symbol}: {str(e)}")
            return None

    def analyze_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phân tích volume profile để xác định vùng tích lũy/phân phối
        
        Args:
            df: DataFrame dữ liệu giá và khối lượng
            
        Returns:
            DataFrame với các chỉ báo volume đã được thêm vào
        """
        try:
            # Thêm các chỉ báo volume
            df['volume_sma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
            
            # Phân loại volume
            df['volume_class'] = pd.qcut(df['volume_ratio'], q=5, labels=['Very Low', 'Low', 'Normal', 'High', 'Very High'])
            
            # Xác định các phiên tích lũy
            df['accumulation'] = (
                (df['close'] > df['open']) & 
                (df['volume'] > df['volume_sma20']) & 
                (df['close'] > df['close'].shift(1))
            )
            
            # Xác định các phiên phân phối
            df['distribution'] = (
                (df['close'] < df['open']) & 
                (df['volume'] > df['volume_sma20']) & 
                (df['close'] < df['close'].shift(1))
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Lỗi khi phân tích volume profile: {str(e)}")
            return df

    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """
        Tìm các mức hỗ trợ và kháng cự
        
        Args:
            df: DataFrame dữ liệu giá
            window: Số ngày để xác định local min/max
            
        Returns:
            Tuple chứa danh sách các mức hỗ trợ và kháng cự
        """
        try:
            # Tìm local minima và maxima
            df['local_min'] = df['low'].rolling(window=window, center=True).min()
            df['local_max'] = df['high'].rolling(window=window, center=True).max()
            
            # Lọc các mức có volume đáng kể
            significant_levels = df[df['volume'] > df['volume_sma20']]
            
            # Tìm các cluster của giá
            support_levels = significant_levels[significant_levels['low'] == significant_levels['local_min']]['low'].unique()
            resistance_levels = significant_levels[significant_levels['high'] == significant_levels['local_max']]['high'].unique()
            
            return list(support_levels), list(resistance_levels)
            
        except Exception as e:
            logging.error(f"Lỗi khi tìm hỗ trợ/kháng cự: {str(e)}")
            return [], []

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán các chỉ báo momentum cho swing trading
        
        Args:
            df: DataFrame dữ liệu giá
            
        Returns:
            DataFrame với các chỉ báo momentum đã được thêm vào
        """
        try:
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # Stochastic
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['close'])
            
            # ADX
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            
            return df
            
        except Exception as e:
            logging.error(f"Lỗi khi tính toán chỉ báo momentum: {str(e)}")
            return df

    def generate_swing_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo tín hiệu swing trading dựa trên các chỉ báo
        
        Args:
            df: DataFrame với các chỉ báo đã được tính toán
            
        Returns:
            DataFrame với các tín hiệu swing trading
        """
        try:
            # Tín hiệu RSI
            df['rsi_signal'] = 0
            df.loc[df['rsi'] < 30, 'rsi_signal'] = 1  # Oversold
            df.loc[df['rsi'] > 70, 'rsi_signal'] = -1  # Overbought
            
            # Tín hiệu Stochastic
            df['stoch_signal'] = 0
            df.loc[(df['stoch_k'] < 20) & (df['stoch_d'] < 20), 'stoch_signal'] = 1
            df.loc[(df['stoch_k'] > 80) & (df['stoch_d'] > 80), 'stoch_signal'] = -1
            
            # Tín hiệu MACD
            df['macd_signal'] = np.where(df['macd'] > 0, 1, -1)
            
            # Tín hiệu Volume
            df['volume_signal'] = 0
            df.loc[df['accumulation'], 'volume_signal'] = 1
            df.loc[df['distribution'], 'volume_signal'] = -1
            
            # Tổng hợp tín hiệu
            df['swing_signal'] = (
                df['rsi_signal'] + 
                df['stoch_signal'] + 
                df['macd_signal'] + 
                df['volume_signal']
            )
            
            # Chuẩn hóa tín hiệu
            df['swing_strength'] = df['swing_signal'].apply(lambda x: 
                'Strong Buy' if x >= 3
                else 'Buy' if x > 0
                else 'Strong Sell' if x <= -3
                else 'Sell' if x < 0
                else 'Neutral'
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Lỗi khi tạo tín hiệu swing: {str(e)}")
            return df

    def process_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Xử lý toàn bộ dữ liệu cho một mã chứng khoán
        
        Args:
            symbol: Mã chứng khoán cần xử lý
            
        Returns:
            Dictionary chứa kết quả phân tích hoặc None nếu có lỗi
        """
        try:
            # 1. Load và tiền xử lý dữ liệu
            df = self.load_daily_data(symbol)
            if df is None:
                return None
                
            # 2. Phân tích volume
            df = self.analyze_volume_profile(df)
            
            # 3. Tìm hỗ trợ/kháng cự
            support_levels, resistance_levels = self.find_support_resistance(df)
            
            # 4. Tính toán chỉ báo momentum
            df = self.calculate_momentum_indicators(df)
            
            # 5. Tạo tín hiệu swing
            df = self.generate_swing_signals(df)
            
            # 6. Lưu kết quả
            current_date = datetime.now().strftime("%Y%m%d")
            
            # Lưu tín hiệu swing
            df.to_csv(self.swing_signals_dir / f"{symbol}_swing_signals_{current_date}.csv")
            
            # Lưu phân tích volume
            volume_analysis = df[['volume', 'volume_sma20', 'volume_ratio', 'volume_class', 'accumulation', 'distribution']]
            volume_analysis.to_csv(self.volume_analysis_dir / f"{symbol}_volume_{current_date}.csv")
            
            # Lưu các mức hỗ trợ/kháng cự
            support_resistance_df = pd.DataFrame({
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            })
            support_resistance_df.to_csv(self.support_resistance_dir / f"{symbol}_levels_{current_date}.csv")
            
            # Lưu chỉ báo momentum
            momentum_df = df[['rsi', 'stoch_k', 'stoch_d', 'macd', 'adx']]
            momentum_df.to_csv(self.momentum_dir / f"{symbol}_momentum_{current_date}.csv")
            
            # Trả về kết quả phân tích
            return {
                'symbol': symbol,
                'last_signal': df['swing_strength'].iloc[-1],
                'last_close': df['close'].iloc[-1],
                'support_levels': support_levels[-3:],  # 3 mức hỗ trợ gần nhất
                'resistance_levels': resistance_levels[:3],  # 3 mức kháng cự gần nhất
                'volume_status': df['volume_class'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'processed_date': current_date
            }
            
        except Exception as e:
            logging.error(f"Lỗi khi xử lý dữ liệu cho {symbol}: {str(e)}")
            return None

def main():
    processor = SwingTradeDataProcessor()
    
    symbols = ['VCB', 'VNM', 'FPT'] 
    
    results = []
    for symbol in symbols:
        result = processor.process_symbol(symbol)
        if result:
            results.append(result)
            logging.info(f"Đã xử lý xong dữ liệu cho {symbol}")
            logging.info(f"Tín hiệu swing cuối cùng: {result['last_signal']}")
            logging.info(f"Các mức hỗ trợ gần nhất: {result['support_levels']}")
            logging.info(f"Các mức kháng cự gần nhất: {result['resistance_levels']}")
            logging.info("------------------------")
    
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(processor.processed_dir / f"swing_analysis_summary_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        logging.info("Đã lưu tổng hợp kết quả phân tích")

if __name__ == "__main__":
    main() 