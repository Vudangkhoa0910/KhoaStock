import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import ta
from datetime import datetime

# Thiết lập thư mục
DATA_DIR = Path('data/collected_data')
OUTPUT_DIR = Path('figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_market_data(symbol='VNM'):
    """Load dữ liệu thị trường từ file"""
    daily_data = pd.read_csv(DATA_DIR / 'daily' / f'{symbol}_daily.csv')
    daily_data['time'] = pd.to_datetime(daily_data['time'])
    daily_data.set_index('time', inplace=True)
    return daily_data

def calculate_technical_indicators(df):
    """Tính toán các chỉ báo kỹ thuật"""
    # Trend Indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
    df['MACD'] = ta.trend.macd(df['close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['close'])
    df['Stoch_K'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['Stoch_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    
    # Volatility Indicators
    df['BB_High'] = ta.volatility.bollinger_hband(df['close'])
    df['BB_Mid'] = ta.volatility.bollinger_mavg(df['close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['close'])
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Calculate BB_%B
    df['BB_%B'] = (df['close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    
    return df

def plot_trend_analysis(df, output_dir):
    """Tạo biểu đồ phân tích xu hướng"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Giá & MA', 'MACD'))
    
    # Candlestick & MA
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='orange')
    ), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Hist'],
        name='MACD Histogram'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=2, col=1)
    
    fig.update_layout(
        title='Phân tích xu hướng với MA và MACD',
        yaxis_title='Giá',
        yaxis2_title='MACD',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    fig.write_image(output_dir / 'trend_analysis.pdf')
    fig.write_html(output_dir / 'trend_analysis.html')

def plot_momentum_analysis(df, output_dir):
    """Tạo biểu đồ phân tích động lượng"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('RSI', 'Stochastic', 'MFI'))
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # Stochastic
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Stoch_K'],
        name='%K',
        line=dict(color='blue')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Stoch_D'],
        name='%D',
        line=dict(color='orange')
    ), row=2, col=1)
    
    # MFI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MFI'],
        name='MFI',
        line=dict(color='purple')
    ), row=3, col=1)
    
    fig.update_layout(
        title='Chỉ báo động lượng',
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    fig.write_image(output_dir / 'momentum_analysis.pdf')
    fig.write_html(output_dir / 'momentum_analysis.html')

def plot_volatility_analysis(df, output_dir):
    """Tạo biểu đồ phân tích biến động"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Bollinger Bands', 'ATR'))
    
    # Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_High'],
        name='Upper BB',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Mid'],
        name='Middle BB',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Low'],
        name='Lower BB',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)
    
    # ATR
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ATR'],
        name='ATR',
        line=dict(color='red')
    ), row=2, col=1)
    
    fig.update_layout(
        title='Chỉ báo biến động',
        yaxis_title='Giá',
        yaxis2_title='ATR',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    fig.write_image(output_dir / 'volatility_analysis.pdf')
    fig.write_html(output_dir / 'volatility_analysis.html')

def plot_correlation_matrix(df, output_dir):
    """Tạo ma trận tương quan"""
    # Chọn các chỉ báo quan trọng
    cols = ['close', 'volume', 'RSI', 'MACD', 'MFI', 'ATR']
    correlation_matrix = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0)
    plt.title('Ma trận tương quan giữa các chỉ báo')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.pdf')
    plt.close()

def plot_composite_indicator(df, output_dir):
    """Tạo biểu đồ chỉ báo tổng hợp"""
    # Tính toán Composite Indicator
    weights = {
        'RSI': 0.2,
        'MACD': 0.2,
        'BB_%B': 0.15,
        'Stoch_K': 0.15,
        'MFI': 0.15,
        'ATR': 0.15
    }
    
    # Chuẩn hóa các chỉ báo
    df_norm = df[list(weights.keys())].apply(lambda x: (x - x.mean()) / x.std())
    composite = sum(df_norm[ind] * weight for ind, weight in weights.items())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Giá đóng cửa',
        yaxis='y1'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=composite,
        name='Composite Indicator',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Chỉ báo tổng hợp và giá',
        yaxis=dict(title='Giá', side='left'),
        yaxis2=dict(title='Composite Indicator', side='right', overlaying='y'),
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    fig.write_image(output_dir / 'composite_indicator.pdf')
    fig.write_html(output_dir / 'composite_indicator.html')

def plot_performance_analysis(df, output_dir):
    """Tạo biểu đồ phân tích hiệu quả"""
    # Tính toán lợi nhuận
    returns = df['close'].pct_change()
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Lợi nhuận tích lũy', 'Drawdown'))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=cum_returns,
        name='Lợi nhuận tích lũy',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=drawdowns,
        name='Drawdown',
        line=dict(color='red')
    ), row=2, col=1)
    
    fig.update_layout(
        title='Phân tích hiệu quả',
        yaxis_title='Lợi nhuận tích lũy',
        yaxis2_title='Drawdown',
        height=800
    )
    
    fig.write_image(output_dir / 'performance_analysis.pdf')
    fig.write_html(output_dir / 'performance_analysis.html')

def plot_risk_analysis(df, output_dir):
    """Tạo biểu đồ phân tích rủi ro"""
    returns = df['close'].pct_change().dropna()
    
    # Tính VaR và CVaR
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        name='Phân phối lợi nhuận',
        nbinsx=50,
        opacity=0.7
    ))
    
    fig.add_vline(x=var_95,
                  line_dash="dash",
                  line_color="red",
                  annotation_text="VaR 95%")
    
    fig.add_vline(x=var_99,
                  line_dash="dash",
                  line_color="darkred",
                  annotation_text="VaR 99%")
    
    fig.update_layout(
        title='Phân phối lợi nhuận và VaR',
        xaxis_title='Lợi nhuận',
        yaxis_title='Tần suất',
        showlegend=True,
        height=600
    )
    
    fig.write_image(output_dir / 'risk_analysis.pdf')
    fig.write_html(output_dir / 'risk_analysis.html')

def plot_swing_trading_ui(df, output_dir):
    """Tạo biểu đồ giao diện Swing Trading"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.5, 0.25, 0.25],
                       subplot_titles=('Biểu đồ nến & EMA', 'MACD', 'RSI'))
    
    # Candlestick & EMA
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_12'],
        name='EMA 12',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_26'],
        name='EMA 26',
        line=dict(color='orange')
    ), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Hist'],
        name='MACD Histogram'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title='Giao diện Swing Trading',
        yaxis_title='Giá',
        yaxis2_title='MACD',
        yaxis3_title='RSI',
        xaxis_rangeslider_visible=False,
        height=1000,
        template='plotly_white'
    )
    
    fig.write_image(output_dir / 'swing_trading_ui.pdf')
    fig.write_html(output_dir / 'swing_trading_ui.html')

def plot_macd_signals(df, output_dir):
    """Tạo biểu đồ tín hiệu MACD với điểm mua/bán"""
    # Tính tín hiệu mua/bán dựa trên MACD
    df['MACD_Signal_Cross'] = np.where(
        (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)),
        'BUY',
        np.where(
            (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)),
            'SELL',
            'HOLD'
        )
    )
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3],
                       subplot_titles=('Biểu đồ nến & Tín hiệu', 'MACD'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Thêm điểm mua
    buy_signals = df[df['MACD_Signal_Cross'] == 'BUY']
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['low'] * 0.99,  # Đặt điểm dưới nến một chút
        mode='markers',
        marker=dict(symbol='triangle-up', size=15, color='green'),
        name='Tín hiệu MUA'
    ), row=1, col=1)
    
    # Thêm điểm bán
    sell_signals = df[df['MACD_Signal_Cross'] == 'SELL']
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['high'] * 1.01,  # Đặt điểm trên nến một chút
        mode='markers',
        marker=dict(symbol='triangle-down', size=15, color='red'),
        name='Tín hiệu BÁN'
    ), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Hist'],
        name='MACD Histogram'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=2, col=1)
    
    fig.update_layout(
        title='Biểu đồ nến với tín hiệu MACD và điểm mua/bán',
        yaxis_title='Giá',
        yaxis2_title='MACD',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_white'
    )
    
    fig.write_image(output_dir / 'macd_signal_chart.pdf')
    fig.write_html(output_dir / 'macd_signal_chart.html')

def plot_rsi_volume(df, output_dir):
    """Tạo biểu đồ RSI và khối lượng giao dịch"""
    # Tính SMA của volume
    df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03,
                       row_heights=[0.4, 0.3, 0.3],
                       subplot_titles=('Biểu đồ nến', 'RSI', 'Khối lượng giao dịch'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green' 
              for index, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Khối lượng',
        marker_color=colors
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_SMA_20'],
        name='SMA 20 của khối lượng',
        line=dict(color='blue')
    ), row=3, col=1)
    
    fig.update_layout(
        title='RSI và khối lượng giao dịch',
        yaxis_title='Giá',
        yaxis2_title='RSI',
        yaxis3_title='Khối lượng',
        xaxis_rangeslider_visible=False,
        height=1000,
        template='plotly_white'
    )
    
    fig.write_image(output_dir / 'rsi_volume_chart.pdf')
    fig.write_html(output_dir / 'rsi_volume_chart.html')

def plot_strategy_comparison(df, output_dir):
    """Tạo biểu đồ so sánh hiệu quả các chiến lược"""
    # Tính toán lợi nhuận Buy & Hold
    buy_hold_returns = (df['close'][-1] - df['close'][0]) / df['close'][0] * 100
    
    # Tính toán lợi nhuận Swing Trading (giả lập dựa trên tín hiệu MACD)
    df['MACD_Signal_Cross'] = np.where(
        (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)),
        1,  # Buy
        np.where(
            (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)),
            -1,  # Sell
            0  # Hold
        )
    )
    
    # Tính lợi nhuận cho mỗi giao dịch
    df['Returns'] = df['close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['MACD_Signal_Cross'].shift(1)
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    swing_returns = (df['Cumulative_Strategy_Returns'][-1] - 1) * 100
    
    # Tạo biểu đồ so sánh
    fig = go.Figure()
    
    # Thêm Buy & Hold
    fig.add_trace(go.Bar(
        x=['Buy & Hold'],
        y=[buy_hold_returns],
        name='Buy & Hold',
        marker_color='blue'
    ))
    
    # Thêm Swing Trading
    fig.add_trace(go.Bar(
        x=['Swing Trading'],
        y=[swing_returns],
        name='Swing Trading',
        marker_color='green'
    ))
    
    fig.update_layout(
        title='So sánh hiệu quả các chiến lược đầu tư',
        yaxis_title='Lợi nhuận (%)',
        showlegend=True,
        template='plotly_white',
        height=500
    )
    
    fig.write_image(output_dir / 'strategy_comparison.pdf')
    fig.write_html(output_dir / 'strategy_comparison.html')

def main():
    # Load dữ liệu
    df = load_market_data('VNM')  # Sử dụng VNM làm ví dụ
    df = calculate_technical_indicators(df)
    
    # Tạo các biểu đồ cho luận văn
    plot_swing_trading_ui(df, OUTPUT_DIR)
    plot_macd_signals(df, OUTPUT_DIR)
    plot_rsi_volume(df, OUTPUT_DIR)
    plot_strategy_comparison(df, OUTPUT_DIR)
    
    # Tạo các biểu đồ phân tích khác
    plot_trend_analysis(df, OUTPUT_DIR)
    plot_momentum_analysis(df, OUTPUT_DIR)
    plot_volatility_analysis(df, OUTPUT_DIR)
    plot_correlation_matrix(df, OUTPUT_DIR)
    plot_composite_indicator(df, OUTPUT_DIR)
    plot_performance_analysis(df, OUTPUT_DIR)
    plot_risk_analysis(df, OUTPUT_DIR)

if __name__ == '__main__':
    main() 