import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create results directory structure
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
STATS_DIR = os.path.join(RESULTS_DIR, 'statistics')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

for dir_path in [RESULTS_DIR, FIGURES_DIR, STATS_DIR, TABLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Load data
def load_daily_data(symbol):
    df = pd.read_csv(f'data/collected_data/daily/{symbol}_daily.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

def load_fundamental_data(symbol):
    df = pd.read_csv(f'data/collected_data/fundamental/{symbol}_ratios.csv', header=1)
    return df

def load_news_data(symbol):
    return pd.read_csv(f'data/collected_data/news/{symbol}_news.csv')

# Load data for all symbols
symbols = ['FPT', 'VNM', 'VCB']
daily_data = {symbol: load_daily_data(symbol) for symbol in symbols}
fundamental_data = {symbol: load_fundamental_data(symbol) for symbol in symbols}

# 1. Price and Volume Analysis
def plot_price_volume(symbol):
    df = daily_data[symbol]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=(f'{symbol} - Giá và Khối lượng giao dịch', ''))
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text=f"Biểu đồ giá và khối lượng giao dịch {symbol}",
        xaxis_rangeslider_visible=False
    )
    
    fig.write_image(os.path.join(FIGURES_DIR, f'{symbol}_price_volume.png'))

# 2. Fundamental Metrics Comparison
def plot_fundamental_comparison():
    metrics = ['ROE (%)', 'P/E', 'EPS (VND)']
    latest_data = {}
    
    for symbol in symbols:
        df = fundamental_data[symbol]
        latest = df.iloc[0]  # Get most recent data
        latest_data[symbol] = [
            float(latest['ROE (%)']),
            float(latest['P/E']),
            float(latest['EPS (VND)'])/1000  # Convert to thousands
        ]
    
    df_comparison = pd.DataFrame(latest_data, index=metrics)
    
    # Save comparison data
    df_comparison.to_csv(os.path.join(TABLES_DIR, 'fundamental_comparison.csv'))
    
    plt.figure(figsize=(12, 6))
    ax = df_comparison.plot(kind='bar')
    plt.title('So sánh chỉ số cơ bản (Q1/2025)')
    plt.xlabel('Chỉ số')
    plt.ylabel('Giá trị')
    plt.legend(title='Mã cổ phiếu')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fundamental_comparison.png'))
    plt.close()

# 3. ROE Trend Analysis
def plot_roe_trend():
    plt.figure(figsize=(12, 6))
    
    roe_trends = {}
    for symbol in symbols:
        df = fundamental_data[symbol]
        df = df.iloc[:8]  # Last 8 quarters
        roe_values = df['ROE (%)'].astype(float)
        roe_trends[symbol] = roe_values
        plt.plot(range(len(df)), roe_values, marker='o', label=symbol)
    
    # Save ROE trends data
    pd.DataFrame(roe_trends).to_csv(os.path.join(TABLES_DIR, 'roe_trends.csv'))
    
    plt.title('Xu hướng ROE theo quý')
    plt.xlabel('Quý (Q1/2023 - Q1/2025)')
    plt.ylabel('ROE (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(8), ['Q1/23', 'Q2/23', 'Q3/23', 'Q4/23', 
                          'Q1/24', 'Q2/24', 'Q3/24', 'Q4/24'])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'roe_trend.png'))
    plt.close()

# 4. Trading Statistics
def calculate_trading_stats(symbol):
    df = daily_data[symbol]
    df['returns'] = df['close'].pct_change()
    
    stats = {
        'Giá trung bình': df['close'].mean(),
        'Khối lượng TB/phiên': df['volume'].mean(),
        'Biến động giá TB (%)': df['returns'].std() * 100,
        'Giá cao nhất': df['high'].max(),
        'Giá thấp nhất': df['low'].min(),
        'Số phiên tăng giá': len(df[df['returns'] > 0]),
        'Số phiên giảm giá': len(df[df['returns'] < 0])
    }
    
    return pd.Series(stats)

# Generate all visualizations and statistics
for symbol in symbols:
    plot_price_volume(symbol)

plot_fundamental_comparison()
plot_roe_trend()

# Create and save trading statistics
trading_stats = pd.DataFrame({symbol: calculate_trading_stats(symbol) for symbol in symbols})
trading_stats.to_csv(os.path.join(TABLES_DIR, 'trading_stats.csv'))

# Create summary report
with open(os.path.join(RESULTS_DIR, 'summary_report.txt'), 'w', encoding='utf-8') as f:
    f.write("KẾT QUẢ PHÂN TÍCH DỮ LIỆU CHỨNG KHOÁN\n")
    f.write("=====================================\n\n")
    f.write("1. THỐNG KÊ GIAO DỊCH\n")
    f.write("-----------------\n")
    f.write(trading_stats.round(2).to_string())
    f.write("\n\n2. ĐƯỜNG DẪN ĐẾN CÁC KẾT QUẢ\n")
    f.write("--------------------------\n")
    f.write(f"- Biểu đồ: {FIGURES_DIR}/\n")
    f.write(f"- Bảng số liệu: {TABLES_DIR}/\n")
    f.write(f"- Thống kê: {STATS_DIR}/\n")

print("\nKết quả đã được lưu vào thư mục:", RESULTS_DIR)
print("- Biểu đồ:", FIGURES_DIR)
print("- Bảng số liệu:", TABLES_DIR)
print("- Thống kê:", STATS_DIR)
print("\nThống kê giao dịch:")
print(trading_stats.round(2)) 