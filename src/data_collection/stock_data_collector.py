# Author: Vu Dang Khoa
from vnstock import Vnstock
from vnstock.explorer.vci import Company, Trading, Quote
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, time as dt_time
import time
import random
from pathlib import Path
import json
import ta
import re
from typing import List, Dict, Optional
import os

class RateLimiter:
    def __init__(self, calls_per_minute=15):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.min_wait = 5
        self.error_mult = 1.5

    def wait(self):
        now = datetime.now()
        self.calls = [t for t in self.calls if (now - t).total_seconds() < 60]
        
        if self.calls:
            last_wait = (now - self.calls[-1]).total_seconds()
            if last_wait < self.min_wait:
                time.sleep(self.min_wait - last_wait)
                
        if len(self.calls) >= self.calls_per_minute:
            wait = 65 - (now - self.calls[0]).total_seconds()
            if wait > 0:
                time.sleep(wait)
                self.calls = []
                
        self.calls.append(now)

    def handle_error(self, msg, retry):
        match = re.search(r'sau (\d+) giây', msg)
        if not match:
            return False
            
        wait = int(match.group(1))
        adj_wait = wait * (self.error_mult ** retry)
        time.sleep(adj_wait)
        self.calls = []
        return True

class StockDataCollector:
    def __init__(self, output_dir: str = "collected_data"):
        self._setup_logging()
        self._init_dirs(output_dir)
        self.trading = Trading()
        self.symbols = self._load_config().get("symbols", ["VCB", "VNM", "FPT"])

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _init_dirs(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.intraday_dir = self.output_dir / "intraday"
        self.market_dir = self.output_dir / "market_data" 
        self.stats_dir = self.output_dir / "trading_stats"
        self.daily_dir = self.output_dir / "daily"
        
        for d in [self.intraday_dir, self.market_dir, self.stats_dir, self.daily_dir]:
            d.mkdir(exist_ok=True)
            
        self.config_file = self.output_dir / "collector_config.json"

    def _load_config(self) -> Dict:
        if not self.config_file.exists():
            return {}
        with open(self.config_file) as f:
            return json.load(f)

    def _save_config(self):
        config = {
            "symbols": self.symbols,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=4)

    def _is_trading_time(self) -> bool:
        now = datetime.now().time()
        morning = dt_time(9, 15) <= now <= dt_time(11, 30)
        afternoon = dt_time(13, 0) <= now <= dt_time(14, 45)
        return morning or afternoon

    def get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=365)
            
            quote = Quote(symbol)
            data = quote.history(
                start_date=start.strftime('%Y-%m-%d'),
                end_date=end.strftime('%Y-%m-%d'),
                interval='1D'
            )
            
            if data is None or data.empty:
                return None
                
            date = datetime.now().strftime("%Y%m%d")
            path = self.daily_dir / f"{symbol}_daily_{date}.csv"
            data.to_csv(path, index=True)
            return data
            
        except Exception as e:
            logging.error(f"Failed to get daily data for {symbol}: {e}")
            return None

    def get_intraday_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self._is_trading_time():
            return None
            
        try:
            quote = Quote(symbol)
            data = quote.intraday(page_size=10000)
            
            if data is None or data.empty:
                return None
                
            date = datetime.now().strftime("%Y%m%d")
            path = self.intraday_dir / f"{symbol}_intraday_{date}.csv"
            data.to_csv(path, index=True)
            return data
            
        except Exception as e:
            logging.error(f"Failed to get intraday data for {symbol}: {e}")
            return None

    def get_market_data(self) -> Optional[pd.DataFrame]:
        try:
            data = self.trading.price_board(self.symbols)
            
            if data is None or data.empty:
                return None
                
            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.market_dir / f"price_board_{date}.csv"
            data.to_csv(path, index=True)
            return data
            
        except Exception as e:
            logging.error(f"Failed to get market data: {e}")
            return None

    def get_trading_stats(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            company = Company(symbol)
            stats = company.trading_stats()
            
            if stats is None or stats.empty:
                return None
                
            date = datetime.now().strftime("%Y%m%d")
            path = self.stats_dir / f"{symbol}_trading_stats_{date}.csv"
            stats.to_csv(path, index=True)
            return stats
            
        except Exception as e:
            logging.error(f"Failed to get trading stats for {symbol}: {e}")
            return None

    def collect_all(self):
        for sym in self.symbols:
            self.get_daily_data(sym)
            time.sleep(5)
            
        for sym in self.symbols:
            self.get_intraday_data(sym)
            time.sleep(5)
            
        self.get_market_data()
        time.sleep(5)
        
        for sym in self.symbols:
            self.get_trading_stats(sym)
            time.sleep(5)

    def add_symbols(self, new_syms: List[str]):
        self.symbols.extend([s for s in new_syms if s not in self.symbols])
        self._save_config()

    def remove_symbols(self, syms: List[str]):
        self.symbols = [s for s in self.symbols if s not in syms]
        self._save_config()

    def get_symbols(self) -> List[str]:
        return self.symbols.copy()

def main():
    collector = StockDataCollector()
    collector.collect_all()

if __name__ == "__main__":
    main()