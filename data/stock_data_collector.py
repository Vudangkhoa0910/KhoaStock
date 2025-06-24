from vnstock import Vnstock
from vnstock.explorer.vci import Company, Trading, Quote  # Import thêm Quote
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, time as dt_time
import time
import random
from pathlib import Path
import json
import ta  # Thư viện technical analysis
import re
from typing import List, Dict, Optional
import os

# Thiết lập logging
logging.basicConfig(
    filename='stock_collector.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RateLimiter:
    """Quản lý rate limit cho API calls"""
    def __init__(self, calls_per_minute=15):  # Giảm xuống 15 calls/phút
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.waiting = False
        self.min_wait_time = 5  # Thời gian chờ tối thiểu giữa các calls
        self.error_wait_multiplier = 1.5  # Hệ số tăng thời gian chờ khi gặp lỗi

    def wait_if_needed(self):
        """Kiểm tra và đợi nếu đã vượt quá rate limit"""
        now = datetime.now()
        # Xóa các calls cũ hơn 1 phút
        self.calls = [call_time for call_time in self.calls 
                     if (now - call_time).total_seconds() < 60]
        
        # Luôn đợi ít nhất min_wait_time giây giữa các calls
        if self.calls:
            last_call_wait = (now - self.calls[-1]).total_seconds()
            if last_call_wait < self.min_wait_time:
                time.sleep(self.min_wait_time - last_call_wait)
        
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 65 - (now - self.calls[0]).total_seconds()  # Thêm 5 giây buffer
            if wait_time > 0:
                logging.info(f"Đợi {wait_time:.1f} giây do rate limit...")
                time.sleep(wait_time)
                self.calls = []  # Reset sau khi đợi
            
        self.calls.append(datetime.now())  # Cập nhật thời gian sau khi đợi

    def handle_rate_limit_error(self, error_message, retry_count):
        """Xử lý khi gặp lỗi rate limit"""
        # Tìm thời gian cần đợi từ thông báo lỗi
        match = re.search(r'sau (\d+) giây', error_message)
        if match:
            wait_time = int(match.group(1))
            # Tăng thời gian đợi theo số lần retry
            adjusted_wait = wait_time * (self.error_wait_multiplier ** retry_count)
            logging.info(f"Rate limit exceeded. Đợi {adjusted_wait:.1f} giây...")
            time.sleep(adjusted_wait)
            self.calls = []  # Reset sau khi đợi
            return True
        return False

class StockDataCollector:
    def __init__(self, output_dir: str = "collected_data"):
        """
        Khởi tạo collector với các thiết lập cơ bản
        
        Args:
            output_dir: Thư mục lưu trữ dữ liệu
        """
        # Thiết lập logging
        self.setup_logging()
        
        # Khởi tạo thư mục lưu trữ
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tạo các thư mục con
        self.intraday_dir = self.output_dir / "intraday"
        self.market_data_dir = self.output_dir / "market_data"
        self.trading_stats_dir = self.output_dir / "trading_stats"
        self.daily_dir = self.output_dir / "daily"  # Thêm thư mục daily
        
        for dir_path in [
            self.intraday_dir, 
            self.market_data_dir, 
            self.trading_stats_dir,
            self.daily_dir
        ]:
            dir_path.mkdir(exist_ok=True)
            
        # Khởi tạo các đối tượng API
        self.trading = Trading()
        
        # Load cấu hình symbols nếu có
        self.config_file = self.output_dir / "collector_config.json"
        self.symbols = self.load_config().get("symbols", ["VCB", "VNM", "FPT"])
        
    def setup_logging(self):
        """Thiết lập logging với format chuẩn"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def load_config(self) -> Dict:
        """Load cấu hình từ file json"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_config(self):
        """Lưu cấu hình hiện tại"""
        config = {
            "symbols": self.symbols,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
    def is_trading_hours(self) -> bool:
        """Kiểm tra xem có phải giờ giao dịch không"""
        now = datetime.now().time()
        morning_session = dt_time(9, 15) <= now <= dt_time(11, 30)
        afternoon_session = dt_time(13, 0) <= now <= dt_time(14, 45)
        return morning_session or afternoon_session
        
    def collect_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Thu thập dữ liệu giao dịch theo ngày"""
        try:
            logging.info(f"Đang thu thập dữ liệu ngày cho {symbol}...")
            
            # Tính toán khoảng thời gian (1 năm gần nhất)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Khởi tạo Quote với symbol
            quote = Quote(symbol)
            
             daily_data = quote.history(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1D'
            )
            
            if daily_data is not None and not daily_data.empty:
                current_date = datetime.now().strftime("%Y%m%d")
                file_path = self.daily_dir / f"{symbol}_daily_{current_date}.csv"
                daily_data.to_csv(file_path, index=True)
                logging.info(f"Đã lưu dữ liệu ngày vào {file_path}")
                return daily_data
                
            logging.warning(f"Không có dữ liệu ngày cho {symbol}")
            return None
            
        except Exception as e:
            logging.error(f"Lỗi khi thu thập dữ liệu ngày cho {symbol}: {str(e)}")
            return None
        
    def collect_intraday_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Thu thập dữ liệu intraday cho một mã chứng khoán"""
        try:
            if not self.is_trading_hours():
                logging.warning(
                    "Ngoài giờ giao dịch. Dữ liệu intraday chỉ khả dụng trong các khung giờ:\n"
                    "- Sáng: 9:15 - 11:30\n"
                    "- Chiều: 13:00 - 14:45"
                )
                return None
                
            logging.info(f"Đang thu thập dữ liệu intraday cho {symbol}...")
            
            quote = Quote(symbol)
            intraday_data = quote.intraday(page_size=10000)
            
            if intraday_data is not None and not intraday_data.empty:
                current_date = datetime.now().strftime("%Y%m%d")
                file_path = self.intraday_dir / f"{symbol}_intraday_{current_date}.csv"
                intraday_data.to_csv(file_path, index=True)
                logging.info(f"Đã lưu dữ liệu intraday vào {file_path}")
                
                logging.info(f"\nThông tin dữ liệu intraday {symbol}:")
                logging.info(f"Số lượng records: {len(intraday_data)}")
                return intraday_data
                
            logging.warning(f"Không có dữ liệu intraday cho {symbol}")
            return None
            
        except Exception as e:
            if "chuẩn bị phiên mới" in str(e):
                logging.error(
                    "Dữ liệu intraday không khả dụng ngoài giờ giao dịch.\n"
                    "Giờ giao dịch:\n"
                    "- Sáng: 9:15 - 11:30\n"
                    "- Chiều: 13:00 - 14:45"
                )
            else:
                logging.error(f"Lỗi khi thu thập dữ liệu intraday cho {symbol}: {str(e)}")
            return None
            
    def collect_market_data(self) -> Optional[pd.DataFrame]:
        """Thu thập dữ liệu bảng giá cho danh sách mã"""
        try:
            logging.info("Đang thu thập dữ liệu bảng giá...")
            
            price_board = self.trading.price_board(self.symbols)
            
            if price_board is not None and not price_board.empty:
                current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = self.market_data_dir / f"price_board_{current_date}.csv"
                price_board.to_csv(file_path, index=True)
                logging.info(f"Đã lưu bảng giá vào {file_path}")
                return price_board
                
            logging.warning("Không có dữ liệu bảng giá")
            return None
            
        except Exception as e:
            logging.error(f"Lỗi khi thu thập dữ liệu bảng giá: {str(e)}")
            return None
            
    def collect_trading_stats(self, symbol: str) -> Optional[pd.DataFrame]:
        """Thu thập thống kê giao dịch cho một mã"""
        try:
            logging.info(f"Đang thu thập thống kê giao dịch cho {symbol}...")
            
            company = Company(symbol)
            trading_stats = company.trading_stats()
            
            if trading_stats is not None and not trading_stats.empty:
                current_date = datetime.now().strftime("%Y%m%d")
                file_path = self.trading_stats_dir / f"{symbol}_trading_stats_{current_date}.csv"
                trading_stats.to_csv(file_path, index=True)
                logging.info(f"Đã lưu thống kê giao dịch vào {file_path}")
                return trading_stats
                
            logging.warning(f"Không có dữ liệu thống kê giao dịch cho {symbol}")
            return None
            
        except Exception as e:
            logging.error(f"Lỗi khi thu thập thống kê giao dịch cho {symbol}: {str(e)}")
            return None
            
    def collect_all_data(self):
        """Thu thập toàn bộ dữ liệu cho tất cả các mã"""
        # 1. Thu thập dữ liệu ngày
        for symbol in self.symbols:
            self.collect_daily_data(symbol)
            time.sleep(5)  # Đợi giữa các lần gọi API
        
        # 2. Thu thập dữ liệu intraday
        for symbol in self.symbols:
            self.collect_intraday_data(symbol)
            time.sleep(5)
            
        # 3. Thu thập bảng giá
        self.collect_market_data()
        time.sleep(5)
        
        # 4. Thu thập thống kê giao dịch
        for symbol in self.symbols:
            self.collect_trading_stats(symbol)
            time.sleep(5)
            
    def add_symbols(self, new_symbols: List[str]):
        """Thêm các mã chứng khoán mới vào danh sách theo dõi"""
        self.symbols.extend([s for s in new_symbols if s not in self.symbols])
        self.save_config()
        
    def remove_symbols(self, symbols_to_remove: List[str]):
        """Xóa các mã chứng khoán khỏi danh sách theo dõi"""
        self.symbols = [s for s in self.symbols if s not in symbols_to_remove]
        self.save_config()
        
    def get_symbols(self) -> List[str]:
        """Lấy danh sách các mã đang theo dõi"""
        return self.symbols.copy()

def main():
    # Khởi tạo collector
    collector = StockDataCollector()
    
    # Thu thập dữ liệu
    collector.collect_all_data()

if __name__ == "__main__":
    main() 