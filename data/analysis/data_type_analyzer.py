import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys
from datetime import datetime

class DataTypeAnalyzer:
    def __init__(self, data_dir: str = None):
        """Khởi tạo analyzer với đường dẫn đến dữ liệu"""
        self.setup_logging()
        
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent
        else:
            self.data_dir = Path(data_dir)
            
    def setup_logging(self):
        """Thiết lập logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def get_detailed_dtype(self, series: pd.Series) -> Dict:
        """Phân tích chi tiết về kiểu dữ liệu của một cột"""
        dtype_info = {
            'pandas_dtype': str(series.dtype),
            'numpy_dtype': str(series.to_numpy().dtype),
            'python_type': str(type(series.iloc[0]).__name__) if len(series) > 0 else None,
            'memory_usage': series.memory_usage(deep=True) / 1024,  # KB
            'is_numeric': pd.api.types.is_numeric_dtype(series),
            'is_integer': pd.api.types.is_integer_dtype(series),
            'is_float': pd.api.types.is_float_dtype(series),
            'is_datetime': pd.api.types.is_datetime64_any_dtype(series),
            'is_categorical': pd.api.types.is_categorical_dtype(series),
            'is_object': pd.api.types.is_object_dtype(series),
            'has_nulls': series.isnull().any(),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique()
        }
        
        # Thêm thông tin về khoảng giá trị cho dữ liệu số
        if dtype_info['is_numeric']:
            dtype_info.update({
                'min': series.min(),
                'max': series.max(),
                'mean': series.mean() if dtype_info['is_float'] else None,
                'std': series.std() if dtype_info['is_float'] else None
            })
            
        # Thêm thông tin về độ dài cho chuỗi
        if dtype_info['is_object']:
            str_lengths = series.str.len()
            dtype_info.update({
                'min_length': str_lengths.min(),
                'max_length': str_lengths.max(),
                'avg_length': str_lengths.mean()
            })
            
        return dtype_info
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Phân tích kiểu dữ liệu của tất cả các cột trong DataFrame"""
        column_types = {}
        for column in df.columns:
            column_types[column] = self.get_detailed_dtype(df[column])
        return column_types
    
    def analyze_file(self, file_path: str) -> Dict:
        """Phân tích kiểu dữ liệu từ file CSV"""
        try:
            df = pd.read_csv(file_path)
            return {
                'file_name': Path(file_path).name,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024,  # KB
                'column_types': self.analyze_dataframe(df)
            }
        except Exception as e:
            logging.error(f"Lỗi khi phân tích file {file_path}: {str(e)}")
            return {}
            
    def generate_type_report(self, symbol: str) -> Dict:
        """Tạo báo cáo chi tiết về kiểu dữ liệu cho một mã chứng khoán"""
        reports = {}
        
        # Phân tích dữ liệu giao dịch hàng ngày
        daily_file = self.data_dir / "collected_data" / "daily" / f"{symbol}_daily.csv"
        if daily_file.exists():
            reports['daily_data'] = self.analyze_file(daily_file)
            
        # Phân tích dữ liệu tín hiệu
        signal_file = self.data_dir / "processed_data" / "swing_signals" / f"{symbol}_swing_signals.csv"
        if signal_file.exists():
            reports['signal_data'] = self.analyze_file(signal_file)
            
        return reports
    
    def print_type_analysis(self, analysis: Dict):
        """In kết quả phân tích kiểu dữ liệu theo định dạng dễ đọc"""
        for data_type, data_info in analysis.items():
            print(f"\n{'='*50}")
            print(f"Phân tích {data_type}:")
            print(f"{'='*50}")
            print(f"File: {data_info['file_name']}")
            print(f"Số dòng: {data_info['total_rows']:,}")
            print(f"Số cột: {data_info['total_columns']}")
            print(f"Dung lượng: {data_info['memory_usage']:.2f} KB")
            
            print("\nChi tiết từng cột:")
            for col, col_info in data_info['column_types'].items():
                print(f"\n{'-'*30}")
                print(f"Cột: {col}")
                print(f"Kiểu dữ liệu Pandas: {col_info['pandas_dtype']}")
                print(f"Kiểu dữ liệu Numpy: {col_info['numpy_dtype']}")
                print(f"Kiểu dữ liệu Python: {col_info['python_type']}")
                print(f"Dung lượng: {col_info['memory_usage']:.2f} KB")
                print(f"Số giá trị null: {col_info['null_count']}")
                print(f"Số giá trị unique: {col_info['unique_count']}")
                
                if col_info['is_numeric']:
                    print(f"Giá trị min: {col_info['min']}")
                    print(f"Giá trị max: {col_info['max']}")
                    if col_info['is_float']:
                        print(f"Giá trị trung bình: {col_info['mean']:.2f}")
                        print(f"Độ lệch chuẩn: {col_info['std']:.2f}")
                
                if col_info['is_object']:
                    print(f"Độ dài min: {col_info['min_length']}")
                    print(f"Độ dài max: {col_info['max_length']}")
                    print(f"Độ dài trung bình: {col_info['avg_length']:.2f}")

def main():
    analyzer = DataTypeAnalyzer()
    
    # Phân tích một số mã chứng khoán mẫu
    symbols = ['FPT', 'VNM', 'VCB']
    for symbol in symbols:
        print(f"\nPhân tích kiểu dữ liệu cho mã {symbol}")
        analysis = analyzer.generate_type_report(symbol)
        analyzer.print_type_analysis(analysis)

if __name__ == "__main__":
    main() 