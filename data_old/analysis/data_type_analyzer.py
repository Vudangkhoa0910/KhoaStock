# Author: Vu Dang Khoa
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys
from datetime import datetime

class DataTypeAnalyzer:
    def __init__(self, data_dir: str = None):
        self._setup_base()
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent

    def _setup_base(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _get_type_info(self, series: pd.Series) -> Dict:
        base_info = self._get_base_info(series)
        if base_info['is_numeric']:
            base_info.update(self._get_numeric_info(series))
        if base_info['is_object']:
            base_info.update(self._get_string_info(series))
        return base_info

    def _get_base_info(self, series: pd.Series) -> Dict:
        return {
            'pandas_dtype': str(series.dtype),
            'numpy_dtype': str(series.to_numpy().dtype),
            'python_type': str(type(series.iloc[0]).__name__) if len(series) > 0 else None,
            'memory_usage': series.memory_usage(deep=True) / 1024,
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

    def _get_numeric_info(self, series: pd.Series) -> Dict:
        info = {'min': series.min(), 'max': series.max()}
        if pd.api.types.is_float_dtype(series):
            info.update({'mean': series.mean(), 'std': series.std()})
        return info

    def _get_string_info(self, series: pd.Series) -> Dict:
        lengths = series.str.len()
        return {
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'avg_length': lengths.mean()
        }

    def analyze_df(self, df: pd.DataFrame) -> Dict[str, Dict]:
        return {col: self._get_type_info(df[col]) for col in df.columns}

    def analyze_csv(self, file_path: str) -> Dict:
        try:
            df = pd.read_csv(file_path)
            return {
                'file': Path(file_path).name,
                'rows': len(df),
                'cols': len(df.columns),
                'size': df.memory_usage(deep=True).sum() / 1024,
                'columns': self.analyze_df(df)
            }
        except Exception as e:
            logging.error(f"Failed to analyze {file_path}: {str(e)}")
            return {}

    def get_report(self, symbol: str) -> Dict:
        report = {}
        daily = self.data_dir / "collected_data" / "daily" / f"{symbol}_daily.csv"
        if daily.exists():
            report['daily'] = self.analyze_csv(daily)
        
        signals = self.data_dir / "processed_data" / "swing_signals" / f"{symbol}_swing_signals.csv"
        if signals.exists():
            report['signals'] = self.analyze_csv(signals)
        return report

    def print_analysis(self, data: Dict):
        for dtype, info in data.items():
            self._print_section(dtype, info)

    def _print_section(self, dtype: str, info: Dict):
        print(f"\n{'='*50}\n{dtype} Analysis:\n{'='*50}")
        print(f"File: {info['file']}")
        print(f"Rows: {info['rows']:,}")
        print(f"Columns: {info['cols']}")
        print(f"Size: {info['size']:.2f} KB")
        
        print("\nColumn Details:")
        for col, col_info in info['columns'].items():
            print(f"\n{'-'*30}")
            print(f"Column: {col}")
            print(f"Pandas Type: {col_info['pandas_dtype']}")
            print(f"Numpy Type: {col_info['numpy_dtype']}")
            print(f"Python Type: {col_info['python_type']}")
            print(f"Size: {col_info['memory_usage']:.2f} KB")
            print(f"Null Count: {col_info['null_count']}")
            print(f"Unique Values: {col_info['unique_count']}")
            
            if col_info['is_numeric']:
                print(f"Min: {col_info['min']}")
                print(f"Max: {col_info['max']}")
                if col_info['is_float']:
                    print(f"Mean: {col_info['mean']:.2f}")
                    print(f"Std Dev: {col_info['std']:.2f}")
            if col_info['is_object']:
                print(f"Min Length: {col_info['min_length']}")
                print(f"Max Length: {col_info['max_length']}")
                print(f"Avg Length: {col_info['avg_length']:.2f}")
def main():
    analyzer = DataTypeAnalyzer()
    symbols = ['FPT', 'VNM', 'VCB']
    for symbol in symbols:
        print(f"\nData Type Analysis for: {symbol}")
        analysis = analyzer.get_report(symbol)
        analyzer.print_analysis(analysis)
if __name__ == "__main__":
    main()