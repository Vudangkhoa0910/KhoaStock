from vnstock import Listing, Quote, Company, Finance, Trading
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
import random
from functools import wraps

# Thiết lập logging
logging.basicConfig(
    filename='vnstock_explorer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Tắt cảnh báo pandas
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)

def handle_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Random delay between 3-7 seconds after successful call
            delay = random.uniform(3, 7)
            time.sleep(delay)
            return result
        except NotImplementedError as e:
            print(f"❌ Chức năng không được hỗ trợ: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Lỗi khi khám phá {func.__name__}: {str(e)}")
            # Longer delay (15-45s) after error
            delay = random.uniform(15, 45)
            time.sleep(delay)
            return None
    return wrapper

class VnstockExplorer:
    def __init__(self, symbol='VCB', source='VCI'):
        """Khởi tạo explorer với một mã chứng khoán và nguồn dữ liệu
        
        Args:
            symbol (str): Mã chứng khoán (mặc định: 'VCB')
            source (str): Nguồn dữ liệu ('VCI', 'TCBS', 'MSN') (mặc định: 'VCI')
        """
        self.symbol = symbol
        self.source = source
        self.listing = Listing(source='VCI')  # Listing chỉ hỗ trợ VCI hoặc MSN
        self.quote = Quote(symbol=symbol, source=source)
        self.company = Company(symbol=symbol, source=source)
        self.finance = Finance(symbol=symbol, source=source)

    @handle_rate_limit
    def explore_listing(self):
        """Khám phá các chức năng liệt kê danh sách"""
        print("\n🔍 KHÁM PHÁ CHỨC NĂNG LISTING\n")
        
        print("1. Danh sách tất cả mã chứng khoán:")
        all_symbols = self.listing.all_symbols()
        print(all_symbols.head() if isinstance(all_symbols, pd.DataFrame) else "Không có dữ liệu")
        
        print("\n2. Danh sách ngành:")
        industries = self.listing.industries_icb()
        print(industries.head() if isinstance(industries, pd.DataFrame) else "Không có dữ liệu")
        
        print("\n3. Danh sách mã theo sàn:")
        by_exchange = self.listing.symbols_by_exchange()
        print(by_exchange.head() if isinstance(by_exchange, pd.DataFrame) else "Không có dữ liệu")

    @handle_rate_limit
    def explore_company_info(self):
        """Khám phá thông tin công ty"""
        print("\n🏢 KHÁM PHÁ THÔNG TIN CÔNG TY\n")
        
        print("1. Thông tin tổng quan công ty:")
        overview = self.company.overview()
        print(overview if isinstance(overview, pd.DataFrame) else "Không có dữ liệu")
        
        print("\n2. Danh sách cổ đông lớn:")
        shareholders = self.company.shareholders()
        print(shareholders.head() if isinstance(shareholders, pd.DataFrame) else "Không có dữ liệu")

    @handle_rate_limit
    def explore_financial_info(self):
        """Khám phá thông tin tài chính"""
        print("\n💰 KHÁM PHÁ THÔNG TIN TÀI CHÍNH\n")
        
        print("1. Chỉ số tài chính:")
        ratios = self.finance.ratio()
        print(ratios.head() if isinstance(ratios, pd.DataFrame) else "Không có dữ liệu")
        
        print("\n2. Báo cáo tài chính:")
        income = self.finance.income_statement()
        print(income.head() if isinstance(income, pd.DataFrame) else "Không có dữ liệu")
        
        print("\n3. Bảng cân đối kế toán:")
        balance = self.finance.balance_sheet()
        print(balance.head() if isinstance(balance, pd.DataFrame) else "Không có dữ liệu")
        
        print("\n4. Báo cáo lưu chuyển tiền tệ:")
        cashflow = self.finance.cash_flow()
        print(cashflow.head() if isinstance(cashflow, pd.DataFrame) else "Không có dữ liệu")

    @handle_rate_limit
    def explore_trading_data(self):
        """Khám phá dữ liệu giao dịch"""
        print("\n📈 KHÁM PHÁ DỮ LIỆU GIAO DỊCH\n")
        
        print("1. Dữ liệu lịch sử:")
        try:
            history = self.quote.history(start="2023-01-01", end="2024-03-31")
            print(history.head() if isinstance(history, pd.DataFrame) else "Không có dữ liệu")
        except Exception as e:
            print(f"❌ Lỗi khi lấy dữ liệu lịch sử: {str(e)}")

    def explore_all(self):
        """Khám phá tất cả chức năng"""
        print("=" * 50)
        print(f"Testing with {self.symbol} (Source: {self.source})")
        print("=" * 50)
        
        print(f"\n🚀 BẮT ĐẦU KHÁM PHÁ VNSTOCK")
        print(f"Mã chứng khoán: {self.symbol}")
        
        self.explore_listing()
        time.sleep(15)  # Delay between major sections
        
        self.explore_company_info()
        time.sleep(15)
        
        self.explore_financial_info()
        time.sleep(15)
        
        self.explore_trading_data()

def main():
    # Test với một số mã phổ biến
    symbols = ['VCB', 'VNM', 'FPT']
    for symbol in symbols:
        explorer = VnstockExplorer(symbol)  # Sử dụng VCI làm nguồn mặc định
        explorer.explore_all()
        time.sleep(45)  # Delay between stocks

if __name__ == "__main__":
    main() 