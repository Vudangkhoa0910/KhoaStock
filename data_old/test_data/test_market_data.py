from vnstock.explorer.vci import Quote, Company, Trading
import pandas as pd
import logging
from pathlib import Path
import time
from datetime import datetime, time as dt_time

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_trading_hours():
    """Kiểm tra xem có phải giờ giao dịch không"""
    now = datetime.now().time()
    morning_session = dt_time(9, 15) <= now <= dt_time(11, 30)
    afternoon_session = dt_time(13, 0) <= now <= dt_time(14, 45)
    return morning_session or afternoon_session

def test_intraday_data(symbol='VCB'):
    """Test lấy dữ liệu intraday theo tick"""
    try:
        if not is_trading_hours():
            logging.warning(
                "Ngoài giờ giao dịch. Dữ liệu intraday chỉ khả dụng trong các khung giờ:\n"
                "- Sáng: 9:15 - 11:30\n"
                "- Chiều: 13:00 - 14:45"
            )
            return
            
        logging.info(f"Đang lấy dữ liệu intraday cho {symbol}...")
        
        # Khởi tạo Quote với symbol
        quote = Quote(symbol)
        
        # Lấy dữ liệu khớp lệnh theo tick
        intraday_data = quote.intraday(
            page_size=10000
        )
        
        if intraday_data is not None and not intraday_data.empty:
            # Tạo thư mục nếu chưa tồn tại
            output_dir = Path('test_data')
            output_dir.mkdir(exist_ok=True)
            
            # Lưu dữ liệu
            file_path = output_dir / f"{symbol}_intraday_test.csv"
            intraday_data.to_csv(file_path, index=True)
            logging.info(f"Đã lưu dữ liệu intraday vào {file_path}")
            
            # Hiển thị thông tin
            logging.info("\nThông tin dữ liệu intraday:")
            logging.info(f"Số lượng records: {len(intraday_data)}")
            logging.info(f"Các cột: {intraday_data.columns.tolist()}")
            logging.info("\nMẫu dữ liệu:")
            print(intraday_data.head())
            
        else:
            logging.warning(f"Không có dữ liệu intraday cho {symbol}")
            
    except Exception as e:
        if "chuẩn bị phiên mới" in str(e):
            logging.error(
                "Dữ liệu intraday không khả dụng ngoài giờ giao dịch.\n"
                "Giờ giao dịch:\n"
                "- Sáng: 9:15 - 11:30\n"
                "- Chiều: 13:00 - 14:45"
            )
        else:
            logging.error(f"Lỗi khi lấy dữ liệu intraday cho {symbol}: {str(e)}")

def test_market_data(symbols=['VCB', 'VNM', 'FPT']):
    """Test lấy dữ liệu bảng giá và thống kê giao dịch"""
    try:
        logging.info("Đang lấy dữ liệu thị trường...")
        
        # Khởi tạo đối tượng Trading
        trading = Trading()
        
        # Tạo thư mục output
        output_dir = Path('test_data')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Lấy bảng giá
        logging.info("Lấy bảng giá...")
        price_board = trading.price_board(symbols)
        
        if price_board is not None and not price_board.empty:
            file_path = output_dir / "price_board_test.csv"
            price_board.to_csv(file_path, index=True)
            logging.info(f"Đã lưu bảng giá vào {file_path}")
            logging.info("\nMẫu bảng giá:")
            print(price_board)
        else:
            logging.warning("Không có dữ liệu bảng giá")
        
        # 2. Lấy thống kê giao dịch cho từng mã
        for symbol in symbols:
            try:
                logging.info(f"\nLấy thống kê giao dịch cho {symbol}...")
                company = Company(symbol)
                trading_stats = company.trading_stats()
                
                if trading_stats is not None and not trading_stats.empty:
                    file_path = output_dir / f"{symbol}_trading_stats_test.csv"
                    trading_stats.to_csv(file_path, index=True)
                    logging.info(f"Đã lưu thống kê giao dịch vào {file_path}")
                    logging.info("\nMẫu thống kê giao dịch:")
                    print(trading_stats.head())
                else:
                    logging.warning(f"Không có dữ liệu thống kê giao dịch cho {symbol}")
                    
            except Exception as e:
                logging.error(f"Lỗi khi lấy thống kê giao dịch cho {symbol}: {str(e)}")
            
            # Đợi giữa các lần gọi API
            time.sleep(5)
            
    except Exception as e:
        logging.error(f"Lỗi khi lấy dữ liệu thị trường: {str(e)}")

def main():
    # Test với một số mã chứng khoán
    symbols_test = ['VCB', 'VNM', 'FPT']
    
    # 1. Test intraday data
    for symbol in symbols_test:
        test_intraday_data(symbol)
        time.sleep(5)  # Đợi 5 giây giữa các lần test
        
    # 2. Test market data
    test_market_data(symbols_test)

if __name__ == "__main__":
    main() 