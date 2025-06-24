import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Phân Tích Chứng Khoán",
    page_icon="📈",
    layout="wide"
)

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLLECTED_DATA_DIR = ROOT_DIR / "data" / "collected_data"

st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1c4b27;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#2e7d32,#1b5e20);
        color: white;
        padding: 1rem;
    }
    div[data-testid="stSidebarNav"] > ul {
        padding-left: 1.5rem;
    }
    .stock-metrics {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pipeline-step {
        border-left: 3px solid #2e7d32;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .market-data {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .price-up {
        color: #2e7d32;
        font-weight: bold;
    }
    .price-down {
        color: #c62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_market_data():
    """Tải dữ liệu thị trường từ file CSV mới nhất"""
    market_data_dir = COLLECTED_DATA_DIR / "market_data"
    market_files = list(market_data_dir.glob("price_board_*.csv"))
    if not market_files:
        return None
    
    latest_file = max(market_files, key=lambda x: x.stat().st_mtime)
    # Đọc file với header=0 để lấy dòng đầu tiên làm tên cột
    df = pd.read_csv(latest_file, header=[0, 1])
    
    # Tạo DataFrame mới với các cột cần thiết
    market_data = pd.DataFrame()
    market_data['symbol'] = df[('listing', 'symbol')]
    market_data['ceiling'] = df[('listing', 'ceiling')].astype(float)
    market_data['floor'] = df[('listing', 'floor')].astype(float)
    market_data['ref_price'] = df[('listing', 'ref_price')].astype(float)
    
    # Thêm các cột từ nhóm match
    market_data['match_price'] = df[('match', 'match_price')].astype(float)
    market_data['accumulated_volume'] = df[('match', 'accumulated_volume')].astype(float)
    market_data['highest'] = df[('match', 'highest')].astype(float)
    market_data['lowest'] = df[('match', 'lowest')].astype(float)
    
    # Thêm thông tin khối ngoại
    market_data['foreign_room'] = df[('match', 'current_room')].astype(float)
    market_data['total_room'] = df[('match', 'total_room')].astype(float)
    market_data['current_holding_ratio'] = (market_data['total_room'] - market_data['foreign_room']) / market_data['total_room']
    
    # Tính KLGD trung bình 2 tuần (giả lập)
    market_data['avg_match_volume_2w'] = market_data['accumulated_volume'] * 0.8  # Giả lập dữ liệu
    
    return market_data

def load_trading_stats(symbol):
    """Tải thống kê giao dịch cho mã chứng khoán"""
    stats_dir = COLLECTED_DATA_DIR / "trading_stats"
    stats_files = list(stats_dir.glob(f"{symbol}_trading_stats_*.csv"))
    if not stats_files:
        return None
    
    latest_file = max(stats_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    return df

def load_technical_data(symbol):
    """Tải dữ liệu chỉ báo kỹ thuật cho mã chứng khoán"""
    tech_file = COLLECTED_DATA_DIR / "technical" / f"{symbol}_technical.csv"
    if not tech_file.exists():
        return None
    
    df = pd.read_csv(tech_file)
    df['time'] = pd.to_datetime(df['time'])
    return df

def format_price(price, ref_price=None):
    """Định dạng giá với màu sắc tương ứng"""
    if ref_price is None:
        return f"{price:,.0f}"
    
    color = "price-up" if price > ref_price else "price-down" if price < ref_price else ""
    return f'<span class="{color}">{price:,.0f}</span>'

# Thanh bên
st.sidebar.title("Quy Trình Phân Tích Chứng Khoán by Vu Dang Khoa")
st.sidebar.markdown("---")

# Chọn mã chứng khoán
STOCK_CODES = ["VCB", "VNM", "FPT"]
selected_stock = st.sidebar.selectbox("Chọn Mã Chứng Khoán", STOCK_CODES)

# Chọn khoảng thời gian
date_range = st.sidebar.date_input(
    "Chọn Khoảng Thời Gian",
    value=(datetime.now() - timedelta(days=365), datetime.now()),
    max_value=datetime.now()
)

# Tiến độ quy trình
pipeline_stages = {
    "Thu Thập Dữ Liệu": ["Dữ Liệu Thị Trường", "Thống Kê Giao Dịch", "Tin Tức", "Cơ Bản"],
    "Xử Lý Dữ Liệu": ["Chỉ Báo Kỹ Thuật", "Phân Tích Khối Lượng", "Nhận Diện Mẫu Hình"],
    "Phân Tích Dữ Liệu": ["Phân Tích Giá", "Tạo Tín Hiệu", "Dự Đoán Mô Hình"]
}

st.sidebar.markdown("---")
st.sidebar.subheader("Tiến Độ Quy Trình")

for stage, substages in pipeline_stages.items():
    st.sidebar.markdown(f"**{stage}**")
    for substage in substages:
        progress = np.random.randint(70, 100)  # Tiến độ mô phỏng
        st.sidebar.text(substage)
        st.sidebar.progress(progress/100)

# Nội dung chính
st.title("Bảng Điều Khiển Phân Tích Chứng Khoán")

# Tải dữ liệu
market_data = load_market_data()
trading_stats = load_trading_stats(selected_stock)
technical_data = load_technical_data(selected_stock)

# Hiển thị dữ liệu thị trường
if market_data is not None:
    stock_data = market_data[market_data['symbol'] == selected_stock].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="market-data">
            <h3>Giá Hiện Tại</h3>
            <p>Giá: {}</p>
            <p>Thay đổi: {} ({:.2f}%)</p>
        </div>
        """.format(
            format_price(stock_data['match_price'], stock_data['ref_price']),
            format_price(stock_data['match_price'] - stock_data['ref_price']),
            (stock_data['match_price'] - stock_data['ref_price']) / stock_data['ref_price'] * 100
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="market-data">
            <h3>Khối Lượng</h3>
            <p>KL Khớp: {:,.0f}</p>
            <p>KLGD TB (2W): {:,.0f}</p>
        </div>
        """.format(
            stock_data['accumulated_volume'],
            stock_data['avg_match_volume_2w']
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="market-data">
            <h3>Giá Tham Chiếu</h3>
            <p>Trần: {}</p>
            <p>Sàn: {}</p>
        </div>
        """.format(
            format_price(stock_data['ceiling']),
            format_price(stock_data['floor'])
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="market-data">
            <h3>Room Nước Ngoài</h3>
            <p>Còn lại: {:,.0f}</p>
            <p>Tỷ lệ sở hữu: {:.2f}%</p>
        </div>
        """.format(
            stock_data['foreign_room'],
            stock_data['current_holding_ratio'] * 100
        ), unsafe_allow_html=True)

# Các tab cho từng giai đoạn
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Thu Thập Dữ Liệu", 
    "⚙️ Xử Lý Dữ Liệu",
    "📊 Phân Tích Dữ Liệu",
    "📈 Phân Tích Tổng Hợp",
    "🏢 Thông Tin & Tin Tức"
])

# Tab Thu Thập Dữ Liệu
with tab1:
    st.header("Giai Đoạn Thu Thập Dữ Liệu")
    
    if technical_data is not None:
        # Hiển thị biểu đồ giá và khối lượng
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          row_heights=[0.7, 0.3])

        # Thêm biểu đồ nến
        fig.add_trace(go.Candlestick(x=technical_data['time'],
                                    open=technical_data['Open'],
                                    high=technical_data['High'],
                                    low=technical_data['Low'],
                                    close=technical_data['Close'],
                                    name="OHLC"), row=1, col=1)

        # Thêm khối lượng
        fig.add_trace(go.Bar(x=technical_data['time'],
                            y=technical_data['Volume'],
                            name="Khối lượng"), row=2, col=1)

        fig.update_layout(
            title=f"{selected_stock} - Biểu Đồ Giá & Khối Lượng",
            yaxis_title="Giá",
            yaxis2_title="Khối lượng",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

# Tab Xử Lý Dữ Liệu
with tab2:
    st.header("Giai Đoạn Xử Lý Dữ Liệu")
    
    if technical_data is not None:
        # Tính toán Price_Change trước khi sử dụng
        technical_data['Price_Change'] = technical_data['Close'].pct_change()
        technical_data['Price_Change_Pct'] = technical_data['Price_Change'] * 100

        # Giải thích phương pháp xử lý dữ liệu
        st.markdown("""
        ### 🔍 Phương Pháp Xử Lý Dữ Liệu
        
        Dữ liệu được xử lý qua nhiều bước để tạo ra các chỉ báo kỹ thuật và phân tích xu hướng:
        
        #### 1️⃣ Xử Lý Dữ liệu Giá
        - **Dữ liệu đầu vào**: OHLC (Open, High, Low, Close) và Volume
        - **Làm sạch dữ liệu**: Loại bỏ các giá trị null và bất thường
        - **Chuẩn hóa**: Điều chỉnh giá theo các sự kiện doanh nghiệp (nếu có)
        
        #### 2️⃣ Chỉ Báo Xu Hướng
        - **SMA (Simple Moving Average)**:
          - SMA20: Trung bình động 20 phiên
          - SMA50: Trung bình động 50 phiên
          - *Công thức*: SMA = (P₁ + P₂ + ... + Pₙ) / n
        
        - **Bollinger Bands**:
          - Dải giữa (BB_Mid): SMA 20 phiên
          - Dải trên (BB_High): BB_Mid + 2 × độ lệch chuẩn
          - Dải dưới (BB_Low): BB_Mid - 2 × độ lệch chuẩn
        
        #### 3️⃣ Chỉ Báo Động Lượng
        - **RSI (Relative Strength Index)**:
          - Khoảng thời gian: 14 phiên
          - Vùng quá mua: > 70
          - Vùng quá bán: < 30
          - *Công thức*: RSI = 100 - (100 / (1 + RS))
            - RS = Trung bình tăng / Trung bình giảm
        
        - **MACD (Moving Average Convergence Divergence)**:
          - MACD Line: EMA(12) - EMA(26)
          - Signal Line: EMA(9) của MACD
          - *Histogram*: MACD - Signal
        
        #### 4️⃣ Chỉ Báo Biến Động
        - **ATR (Average True Range)**:
          - Khoảng thời gian: 14 phiên
          - *True Range* = max(H-L, |H-C_prev|, |L-C_prev|)
        
        - **OBV (On Balance Volume)**:
          - Tích lũy khối lượng theo chiều giá
          - Tăng khi giá đóng cửa tăng
          - Giảm khi giá đóng cửa giảm
        """)

        # Hiển thị các chỉ báo kỹ thuật
        st.markdown("### 📊 Biểu Đồ Phân Tích Kỹ Thuật")
        
        # Tạo tabs cho các nhóm chỉ báo
        tech_tab1, tech_tab2, tech_tab3 = st.tabs([
            "Xu Hướng & Bollinger Bands",
            "Chỉ Báo Động Lượng",
            "Biến Động & Khối Lượng"
        ])
        
        with tech_tab1:
            st.markdown("""
            #### 📈 Phân Tích Xu Hướng và Bollinger Bands
            
            - **Đường MA (Moving Average)**: 
              - MA20 (ngắn hạn) và MA50 (trung hạn) giúp xác định xu hướng
              - Cắt lên: Tín hiệu tăng giá
              - Cắt xuống: Tín hiệu giảm giá
            
            - **Dải Bollinger**:
              - Độ rộng dải thể hiện biến động
              - Giá chạm dải trên: Có thể quá mua
              - Giá chạm dải dưới: Có thể quá bán
            """)
            
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.7, 0.3])

            fig.add_trace(go.Candlestick(x=technical_data['time'],
                                        open=technical_data['Open'],
                                        high=technical_data['High'],
                                        low=technical_data['Low'],
                                        close=technical_data['Close'],
                                        name="Giá"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['SMA_20'],
                                   name="SMA20",
                                   line=dict(color='blue', width=1)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['SMA_50'],
                                   name="SMA50",
                                   line=dict(color='orange', width=1)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['BB_High'],
                                   name="BB Upper",
                                   line=dict(color='gray', dash='dash'),
                                   opacity=0.3), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['BB_Low'],
                                   name="BB Lower",
                                   line=dict(color='gray', dash='dash'),
                                   fill='tonexty',
                                   opacity=0.3), row=1, col=1)
            
            bb_width = ((technical_data['BB_High'] - technical_data['BB_Low']) / 
                       technical_data['BB_Mid'] * 100)
            
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=bb_width,
                                   name="BB Width (%)",
                                   line=dict(color='purple')), row=2, col=1)
            
            fig.update_layout(height=800, title="Phân Tích Xu Hướng và Bollinger Bands")
            st.plotly_chart(fig, use_container_width=True)

        with tech_tab2:
            st.markdown("""
            #### 📊 Phân Tích Động Lượng
            
            - **RSI (Relative Strength Index)**:
              - Đo lường tốc độ và độ lớn của biến động giá
              - RSI > 70: Vùng quá mua
              - RSI < 30: Vùng quá bán
            
            - **MACD**:
              - MACD Line cắt Signal Line lên: Tín hiệu mua
              - MACD Line cắt Signal Line xuống: Tín hiệu bán
              - Histogram thể hiện độ mạnh của xu hướng
            
            - **Stochastic**:
              - %K và %D cho biết vị trí giá trong phạm vi giao dịch
              - Vùng quá mua/bán tương tự RSI
            """)
            
            fig = make_subplots(rows=3, cols=1,
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.4, 0.3, 0.3],
                              subplot_titles=("RSI", "MACD", "Stochastic"))

            # RSI
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['RSI'],
                                   name="RSI",
                                   line=dict(color='blue')), row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['MACD'],
                                   name="MACD",
                                   line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['MACD_Signal'],
                                   name="Signal",
                                   line=dict(color='orange')), row=2, col=1)
            
            # Tính toán MACD Histogram
            macd_hist = technical_data['MACD'] - technical_data['MACD_Signal']
            colors = ['green' if x >= 0 else 'red' for x in macd_hist]
            fig.add_trace(go.Bar(x=technical_data['time'],
                               y=macd_hist,
                               name="MACD Hist",
                               marker_color=colors), row=2, col=1)

            # Stochastic
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['Stoch'],
                                   name="%K",
                                   line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['Stoch_Signal'],
                                   name="%D",
                                   line=dict(color='orange')), row=3, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

            fig.update_layout(height=800, title="Phân Tích Động Lượng")
            st.plotly_chart(fig, use_container_width=True)

        with tech_tab3:
            st.markdown("""
            #### 📉 Phân Tích Biến Động và Khối Lượng
            
            - **ATR (Average True Range)**:
              - Đo lường biến động giá trung bình
              - ATR cao: Biến động mạnh
              - ATR thấp: Biến động yếu
            
            - **Khối lượng**:
              - Khối lượng tăng khi giá tăng: Xác nhận xu hướng tăng
              - Khối lượng tăng khi giá giảm: Xác nhận xu hướng giảm
              - Khối lượng thấp: Thiếu sự quan tâm của thị trường
            
            - **OBV (On Balance Volume)**:
              - Tích lũy khối lượng theo chiều giá
              - OBV tăng cùng giá: Xu hướng tăng mạnh
              - OBV giảm khi giá tăng: Cảnh báo xu hướng yếu
            """)
            
            fig = make_subplots(rows=3, cols=1,
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.4, 0.3, 0.3],
                              subplot_titles=("ATR", "Khối Lượng", "OBV"))

            # ATR
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['ATR'],
                                   name="ATR",
                                   line=dict(color='blue')), row=1, col=1)

            # Sử dụng Price_Change đã được tính toán ở trên
            colors = ['green' if x >= 0 else 'red' for x in technical_data['Price_Change']]
            fig.add_trace(go.Bar(x=technical_data['time'],
                               y=technical_data['Volume'],
                               name="Khối lượng",
                               marker_color=colors), row=2, col=1)

            # OBV
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['OBV'],
                                   name="OBV",
                                   line=dict(color='purple')), row=3, col=1)

            fig.update_layout(height=800, title="Phân Tích Biến Động và Khối Lượng")
            st.plotly_chart(fig, use_container_width=True)

        # Thêm phần giải thích tín hiệu giao dịch
        st.markdown("""
        ### 🎯 Tín Hiệu Giao Dịch
        
        Dựa trên sự kết hợp của các chỉ báo, hệ thống xác định các tín hiệu giao dịch sau:
        
        #### Tín Hiệu Mua:
        1. RSI < 30 (quá bán)
        2. Giá chạm/vượt dải dưới Bollinger
        3. MACD cắt Signal Line từ dưới lên
        4. Stochastic %K cắt %D từ dưới lên trong vùng quá bán
        
        #### Tín Hiệu Bán:
        1. RSI > 70 (quá mua)
        2. Giá chạm/vượt dải trên Bollinger
        3. MACD cắt Signal Line từ trên xuống
        4. Stochastic %K cắt %D từ trên xuống trong vùng quá mua
        
        #### Xác Nhận Xu Hướng:
        - **Xu hướng tăng** được xác nhận khi:
          - Giá trên MA20 và MA50
          - OBV tăng cùng chiều giá
          - Khối lượng tăng trong phiên tăng giá
        
        - **Xu hướng giảm** được xác nhận khi:
          - Giá dưới MA20 và MA50
          - OBV giảm cùng chiều giá
          - Khối lượng tăng trong phiên giảm giá
        """)

# Tab Phân Tích Dữ Liệu
with tab3:
    st.header("Giai Đoạn Phân Tích Dữ Liệu")
    
    if technical_data is not None:
        # Phân tích xu hướng
        latest_data = technical_data.iloc[-1]
        
        # Tính toán các chỉ số xu hướng
        technical_data['Price_Change'] = technical_data['Close'].pct_change()
        technical_data['Price_Change_Pct'] = technical_data['Price_Change'] * 100
        
        # Tính RSI và các dải Bollinger
        latest_rsi_signal = 'Quá mua' if latest_data['RSI'] > 70 else 'Quá bán' if latest_data['RSI'] < 30 else 'Trung tính'
        latest_bb_position = 'Trên dải trên' if latest_data['Close'] > latest_data['BB_High'] else 'Dưới dải dưới' if latest_data['Close'] < latest_data['BB_Low'] else 'Trong dải'
        
        # Hiển thị thông tin xu hướng
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend = "Tăng" if latest_data['SMA_20'] > latest_data['SMA_50'] else "Giảm"
            trend_color = "green" if trend == "Tăng" else "red"
            st.markdown(f"""
            <div class="stock-metrics">
                <h3>Xu Hướng Giá</h3>
                <p>SMA20: {latest_data['SMA_20']:,.2f}</p>
                <p>SMA50: {latest_data['SMA_50']:,.2f}</p>
                <p>Xu hướng: <span style="color: {trend_color}; font-weight: bold;">{trend}</span></p>
                <p>Biến động: {latest_data['ATR']:,.2f} ({(latest_data['ATR']/latest_data['Close']*100):,.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rsi_color = "red" if latest_data['RSI'] > 70 else "green" if latest_data['RSI'] < 30 else "black"
            st.markdown(f"""
            <div class="stock-metrics">
                <h3>Chỉ Báo Động Lượng</h3>
                <p>RSI: <span style="color: {rsi_color}; font-weight: bold;">{latest_data['RSI']:.2f}</span></p>
                <p>MACD: {latest_data['MACD']:.2f}</p>
                <p>MACD Signal: {latest_data['MACD_Signal']:.2f}</p>
                <p>Stochastic: {latest_data['Stoch']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            bb_color = "red" if latest_bb_position == 'Trên dải trên' else "green" if latest_bb_position == 'Dưới dải dưới' else "black"
            st.markdown(f"""
            <div class="stock-metrics">
                <h3>Phân Tích Bollinger Bands</h3>
                <p>Vị trí: <span style="color: {bb_color}; font-weight: bold;">{latest_bb_position}</span></p>
                <p>BB Width: {(latest_data['BB_High'] - latest_data['BB_Low'])/latest_data['BB_Mid']*100:.2f}%</p>
                <p>Khoảng cách từ giá: {abs(latest_data['Close'] - latest_data['BB_Mid'])/latest_data['BB_Mid']*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Biểu đồ xu hướng giá và khối lượng
        st.subheader("Phân Tích Xu Hướng Giá và Khối Lượng")
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25],
                           subplot_titles=("Giá & Xu Hướng", "Khối Lượng", "Momentum"))

        # Biểu đồ giá và MA
        fig.add_trace(go.Candlestick(x=technical_data['time'],
                                    open=technical_data['Open'],
                                    high=technical_data['High'],
                                    low=technical_data['Low'],
                                    close=technical_data['Close'],
                                    name="Giá"), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=technical_data['time'], 
                                y=technical_data['SMA_20'],
                                name="SMA20",
                                line=dict(color='blue', width=1)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=technical_data['time'], 
                                y=technical_data['SMA_50'],
                                name="SMA50",
                                line=dict(color='orange', width=1)), row=1, col=1)
        
        # Thêm Bollinger Bands
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['BB_High'],
                                name="BB Upper",
                                line=dict(color='gray', dash='dash'),
                                opacity=0.3), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['BB_Low'],
                                name="BB Lower",
                                line=dict(color='gray', dash='dash'),
                                fill='tonexty',
                                opacity=0.3), row=1, col=1)

        # Biểu đồ khối lượng với màu sắc theo xu hướng
        colors = ['green' if x >= 0 else 'red' for x in technical_data['Price_Change']]
        fig.add_trace(go.Bar(x=technical_data['time'],
                            y=technical_data['Volume'],
                            name="Khối lượng",
                            marker_color=colors), row=2, col=1)

        # Biểu đồ Momentum (RSI)
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['RSI'],
                                name="RSI",
                                line=dict(color='purple')), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800,
                         showlegend=True,
                         title_text=f"Phân Tích Kỹ Thuật Chi Tiết - {selected_stock}")
        
        st.plotly_chart(fig, use_container_width=True)

        # Phân tích xu hướng theo thời gian
        st.subheader("Phân Tích Xu Hướng Theo Thời Gian")
        
        # Tính toán các chỉ số xu hướng
        periods = [5, 10, 20, 50]
        trend_data = {}
        
        for period in periods:
            changes = technical_data['Close'].pct_change(period) * 100
            if not changes.empty:
                trend_data[f'Thay đổi {period} phiên'] = changes.iloc[-1]
            else:
                trend_data[f'Thay đổi {period} phiên'] = 0
            
        # Hiển thị phân tích xu hướng
        col1, col2 = st.columns(2)
        
        with col1:
            # Biểu đồ xu hướng theo thời gian
            fig = go.Figure()
            
            for period in periods:
                changes = technical_data['Close'].pct_change(period) * 100
                if not changes.empty:
                    fig.add_trace(go.Scatter(x=technical_data['time'],
                                           y=changes,
                                           name=f'Thay đổi {period} phiên',
                                           line=dict(width=1)))
            
            fig.update_layout(title=f"Xu Hướng Theo Thời Gian - {selected_stock}",
                            yaxis_title="Thay đổi (%)",
                            height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hiển thị bảng phân tích xu hướng
            trend_summary = pd.DataFrame({
                'Chỉ số': [
                    'Xu hướng ngắn hạn (5 phiên)',
                    'Xu hướng trung hạn (20 phiên)',
                    'Xu hướng dài hạn (50 phiên)',
                    'Độ biến động (ATR)',
                    'Momentum (RSI)',
                    'Xu hướng MACD'
                ],
                'Trạng thái': [
                    'Tăng' if trend_data['Thay đổi 5 phiên'] > 0 else 'Giảm',
                    'Tăng' if trend_data['Thay đổi 20 phiên'] > 0 else 'Giảm',
                    'Tăng' if trend_data['Thay đổi 50 phiên'] > 0 else 'Giảm',
                    f"{latest_data['ATR']:.2f} ({(latest_data['ATR']/latest_data['Close']*100):.2f}%)",
                    latest_rsi_signal,
                    'Tăng' if latest_data['MACD'] > latest_data['MACD_Signal'] else 'Giảm'
                ],
                'Giá trị': [
                    f"{trend_data['Thay đổi 5 phiên']:.2f}%",
                    f"{trend_data['Thay đổi 20 phiên']:.2f}%",
                    f"{trend_data['Thay đổi 50 phiên']:.2f}%",
                    f"{latest_data['Close']:.2f}",
                    f"{latest_data['RSI']:.2f}",
                    f"{latest_data['MACD']:.2f}"
                ]
            })
            
            st.dataframe(trend_summary,
                        column_config={
                            "Chỉ số": st.column_config.TextColumn("Chỉ số"),
                            "Trạng thái": st.column_config.TextColumn("Trạng thái"),
                            "Giá trị": st.column_config.TextColumn("Giá trị")
                        },
                        hide_index=True)

# Tab Phân Tích Tổng Hợp
with tab4:
    st.header("Phân Tích Tổng Hợp")
    
    if technical_data is not None and trading_stats is not None:
        # Hiển thị thống kê giao dịch
        st.subheader("Thống Kê Giao Dịch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stock-metrics">
                <h3>Thống Kê Giá</h3>
                <p>Cao nhất 52 tuần: {}</p>
                <p>Thấp nhất 52 tuần: {}</p>
                <p>% Thay đổi từ đáy: {:.2f}%</p>
            </div>
            """.format(
                format_price(trading_stats.iloc[0]['high_price_1y']),
                format_price(trading_stats.iloc[0]['low_price_1y']),
                trading_stats.iloc[0]['pct_low_change_1y'] * 100
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stock-metrics">
                <h3>Thống Kê Khối Ngoại</h3>
                <p>Room NN còn lại: {:,.0f}</p>
                <p>Tỷ lệ sở hữu: {:.2f}%</p>
                <p>KLGD NN: {:,.0f}</p>
            </div>
            """.format(
                trading_stats.iloc[0]['foreign_room'],
                trading_stats.iloc[0]['current_holding_ratio'] * 100,
                trading_stats.iloc[0]['foreign_volume']
            ), unsafe_allow_html=True)
        
        # Hiển thị biểu đồ phân tích kỹ thuật tổng hợp
        st.subheader("Phân Tích Kỹ Thuật Tổng Hợp")
        
        # Tính toán các tín hiệu
        signals = pd.DataFrame(index=technical_data.index)
        signals['Signal'] = 'Giữ'
        
        # Tín hiệu từ RSI
        signals.loc[technical_data['RSI'] > 70, 'Signal'] = 'Bán'
        signals.loc[technical_data['RSI'] < 30, 'Signal'] = 'Mua'
        
        # Tín hiệu từ MACD
        signals.loc[(technical_data['MACD'] > technical_data['MACD_Signal']) & 
                   (technical_data['MACD'].shift(1) <= technical_data['MACD_Signal'].shift(1)), 'Signal'] = 'Mua'
        signals.loc[(technical_data['MACD'] < technical_data['MACD_Signal']) & 
                   (technical_data['MACD'].shift(1) >= technical_data['MACD_Signal'].shift(1)), 'Signal'] = 'Bán'
        
        # Vẽ biểu đồ với tín hiệu
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(x=technical_data['time'],
                                    open=technical_data['Open'],
                                    high=technical_data['High'],
                                    low=technical_data['Low'],
                                    close=technical_data['Close'],
                                    name="OHLC"))
        
        # Thêm các đường MA
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['SMA_20'],
                                name="SMA20",
                                line=dict(color='blue')))
        
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['SMA_50'],
                                name="SMA50",
                                line=dict(color='orange')))
        
        # Thêm tín hiệu mua/bán
        buy_signals = signals[signals['Signal'] == 'Mua']
        sell_signals = signals[signals['Signal'] == 'Bán']
        
        fig.add_trace(go.Scatter(
            x=technical_data.loc[buy_signals.index, 'time'],
            y=technical_data.loc[buy_signals.index, 'Low'] * 0.99,
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name='Tín hiệu Mua'
        ))
        
        fig.add_trace(go.Scatter(
            x=technical_data.loc[sell_signals.index, 'time'],
            y=technical_data.loc[sell_signals.index, 'High'] * 1.01,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Tín hiệu Bán'
        ))
        
        fig.update_layout(
            title=f"{selected_stock} - Phân Tích Kỹ Thuật Tổng Hợp",
            yaxis_title="Giá",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab Thông Tin & Tin Tức
with tab5:
    st.header("Thông Tin Công Ty & Tin Tức")
    
    # Đọc dữ liệu công ty
    def load_company_data(symbol):
        try:
            data = {}
            
            # Đọc thông tin tổng quan
            overview_path = COLLECTED_DATA_DIR / "company" / f"{symbol}_overview.csv"
            if overview_path.exists():
                data['overview'] = pd.read_csv(overview_path)
            else:
                st.error(f"Không tìm thấy file thông tin tổng quan: {overview_path}")
                return None
            
            # Đọc thông tin cổ đông
            shareholders_path = COLLECTED_DATA_DIR / "company" / f"{symbol}_shareholders.csv"
            if shareholders_path.exists():
                data['shareholders'] = pd.read_csv(shareholders_path)
            else:
                st.warning(f"Không tìm thấy file thông tin cổ đông: {shareholders_path}")
                data['shareholders'] = pd.DataFrame()
            
            # Đọc thông tin ban lãnh đạo
            officers_path = COLLECTED_DATA_DIR / "company" / f"{symbol}_officers_working.csv"
            if officers_path.exists():
                data['officers'] = pd.read_csv(officers_path)
            else:
                st.warning(f"Không tìm thấy file thông tin ban lãnh đạo: {officers_path}")
                data['officers'] = pd.DataFrame()
            
            return data
            
        except Exception as e:
            st.error(f"Lỗi khi đọc dữ liệu công ty: {str(e)}")
            return None

    def load_news_data(symbol):
        try:
            data = {}
            
            # Đọc tin tức
            news_path = COLLECTED_DATA_DIR / "news" / f"{symbol}_news.csv"
            if news_path.exists():
                data['news'] = pd.read_csv(news_path)
            else:
                st.warning(f"Không tìm thấy file tin tức: {news_path}")
                data['news'] = pd.DataFrame()
            
            # Đọc sự kiện
            events_path = COLLECTED_DATA_DIR / "news" / f"{symbol}_events.csv"
            if events_path.exists():
                data['events'] = pd.read_csv(events_path)
            else:
                st.warning(f"Không tìm thấy file sự kiện: {events_path}")
                data['events'] = pd.DataFrame()
            
            return data
            
        except Exception as e:
            st.error(f"Lỗi khi đọc dữ liệu tin tức: {str(e)}")
            return None

    # Load dữ liệu cho mã được chọn
    company_data = load_company_data(selected_stock)
    news_data = load_news_data(selected_stock)

    if company_data and 'overview' in company_data:
        # Hiển thị thông tin công ty
        st.subheader("Thông Tin Tổng Quan")
        overview = company_data['overview'].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ### {selected_stock} - {overview['icb_name3']}
            
            **Lịch sử phát triển:**
            {overview['history']}
            
            **Giới thiệu công ty:**
            {overview['company_profile']}
            """)
        
        with col2:
            st.markdown(f"""
            ### Thông Tin Cơ Bản
            - **Vốn điều lệ:** {int(overview['charter_capital']):,} VND
            - **Số lượng CP:** {int(overview['issue_share']):,}
            - **Ngành:** {overview['icb_name3']}
            """)

        # Hiển thị cơ cấu cổ đông nếu có
        if 'shareholders' in company_data and not company_data['shareholders'].empty:
            st.subheader("Cơ Cấu Cổ Đông")
            shareholders = company_data['shareholders']
            
            fig = px.pie(shareholders, 
                        values='share_own_percent', 
                        names='share_holder',
                        title=f"Cơ Cấu Cổ Đông {selected_stock}")
            st.plotly_chart(fig, use_container_width=True)

        # Hiển thị ban lãnh đạo nếu có
        if 'officers' in company_data and not company_data['officers'].empty:
            st.subheader("Ban Lãnh Đạo")
            officers = company_data['officers']
            st.dataframe(
                officers[['officer_name', 'officer_position', 'quantity']],
                column_config={
                    "officer_name": "Họ và Tên",
                    "officer_position": "Chức Vụ",
                    "quantity": "Số Lượng CP Sở Hữu"
                },
                hide_index=True
            )

    # Hiển thị tin tức và sự kiện
    if news_data:
        st.subheader("Tin Tức & Sự Kiện")
        
        # Kết hợp tin tức và sự kiện
        timeline_data = []
        
        if 'news' in news_data and not news_data['news'].empty:
            news = news_data['news']
            for _, row in news.iterrows():
                timeline_data.append({
                    'Ngày': pd.to_datetime(row['public_date']),
                    'Nội Dung': row['news_title'],
                    'Loại': 'Tin Tức',
                    'Link': row['news_source_link']
                })
        
        if 'events' in news_data and not news_data['events'].empty:
            events = news_data['events']
            for _, row in events.iterrows():
                timeline_data.append({
                    'Ngày': pd.to_datetime(row['public_date']),
                    'Nội Dung': row['event_title'],
                    'Loại': 'Sự Kiện',
                    'Link': row['source_url'] if pd.notna(row['source_url']) else ''
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('Ngày', ascending=False)
            
            # Hiển thị timeline
            for _, row in timeline_df.head(10).iterrows():
                with st.expander(f"{row['Ngày'].strftime('%d/%m/%Y')} - {row['Nội Dung']}", expanded=False):
                    st.markdown(f"""
                    **Loại:** {row['Loại']}
                    
                    **Link:** [{row['Nội Dung']}]({row['Link']})
                    """)
    else:
        st.warning("Không thể tải dữ liệu tin tức. Vui lòng kiểm tra lại đường dẫn và dữ liệu.") 