# Author: Vu Dang Khoa
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

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="wide"
)

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLLECTED_DATA_DIR = ROOT_DIR / "data" / "collected_data"

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stProgress > div > div > div > div { background-color: #1c4b27; }
    .stAlert > div { 
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#2e7d32,#1b5e20);
        color: white;
        padding: 1rem;
    }
    div[data-testid="stSidebarNav"] > ul { padding-left: 1.5rem; }
    .stock-metrics {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .market-data {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .price-up { color: #2e7d32; font-weight: bold; }
    .price-down { color: #c62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def load_market_data():
    market_data_dir = COLLECTED_DATA_DIR / "market_data"
    market_files = list(market_data_dir.glob("price_board_*.csv"))
    if not market_files:
        return None
    latest_file = max(market_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, header=[0, 1])
    market_data = pd.DataFrame()
    market_data['symbol'] = df[('listing', 'symbol')]
    market_data['ceiling'] = df[('listing', 'ceiling')].astype(float)
    market_data['floor'] = df[('listing', 'floor')].astype(float)
    market_data['ref_price'] = df[('listing', 'ref_price')].astype(float)
    market_data['match_price'] = df[('match', 'match_price')].astype(float)
    market_data['accumulated_volume'] = df[('match', 'accumulated_volume')].astype(float)
    market_data['highest'] = df[('match', 'highest')].astype(float)
    market_data['lowest'] = df[('match', 'lowest')].astype(float)
    market_data['foreign_room'] = df[('match', 'current_room')].astype(float)
    market_data['total_room'] = df[('match', 'total_room')].astype(float)
    market_data['current_holding_ratio'] = (market_data['total_room'] - market_data['foreign_room']) / market_data['total_room']
    market_data['avg_match_volume_2w'] = market_data['accumulated_volume'] * 0.8
    return market_data

def load_trading_stats(symbol):
    stats_dir = COLLECTED_DATA_DIR / "trading_stats"
    stats_files = list(stats_dir.glob(f"{symbol}_trading_stats_*.csv"))
    if not stats_files:
        return None
    latest_file = max(stats_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    return df

def load_technical_data(symbol):
    tech_file = COLLECTED_DATA_DIR / "technical" / f"{symbol}_technical.csv"
    if not tech_file.exists():
        return None
    df = pd.read_csv(tech_file)
    df['time'] = pd.to_datetime(df['time'])
    return df

def format_price(price, ref_price=None):
    if ref_price is None:
        return f"{price:,.0f}"
    color = "price-up" if price > ref_price else "price-down" if price < ref_price else ""
    return f'<span class="{color}">{price:,.0f}</span>'

st.sidebar.title("Stock Analysis Process")
st.sidebar.markdown("---")
STOCK_CODES = ["VCB", "VNM", "FPT"]
selected_stock = st.sidebar.selectbox("Select Stock Symbol", STOCK_CODES)
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=365), datetime.now()),
    max_value=datetime.now()
)
pipeline_stages = {
    "Data Collection": ["Market Data", "Trading Statistics", "News", "Fundamentals"],
    "Data Processing": ["Technical Indicators", "Volume Analysis", "Pattern Recognition"],
    "Data Analysis": ["Price Analysis", "Signal Generation", "Model Prediction"]
}
st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Progress")
for stage, substages in pipeline_stages.items():
    st.sidebar.markdown(f"**{stage}**")
    for substage in substages:
        progress = np.random.randint(70, 100)
        st.sidebar.text(substage)
        st.sidebar.progress(progress/100)
st.title("Stock Analysis Dashboard")
market_data = load_market_data()
trading_stats = load_trading_stats(selected_stock)
technical_data = load_technical_data(selected_stock)
if market_data is not None:
    stock_data = market_data[market_data['symbol'] == selected_stock].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="market-data">
            <h3>Current Price</h3>
            <p>Price: {}</p>
            <p>Change: {} ({:.2f}%)</p>
        </div>
        """.format(
            format_price(stock_data['match_price'], stock_data['ref_price']),
            format_price(stock_data['match_price'] - stock_data['ref_price']),
            (stock_data['match_price'] - stock_data['ref_price']) / stock_data['ref_price'] * 100
        ), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="market-data">
            <h3>Volume</h3>
            <p>Matched: {:,.0f}</p>
            <p>Avg (2W): {:,.0f}</p>
        </div>
        """.format(
            stock_data['accumulated_volume'],
            stock_data['avg_match_volume_2w']
        ), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="market-data">
            <h3>Reference Price</h3>
            <p>Ceiling: {}</p>
            <p>Floor: {}</p>
        </div>
        """.format(
            format_price(stock_data['ceiling']),
            format_price(stock_data['floor'])
        ), unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="market-data">
            <h3>Foreign Room</h3>
            <p>Available: {:,.0f}</p>
            <p>Ownership: {:.2f}%</p>
        </div>
        """.format(
            stock_data['foreign_room'],
            stock_data['current_holding_ratio'] * 100
        ), unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Collection",
    "Data Processing",
    "Data Analysis",
    "Comprehensive Analysis",
    "Information & News"
])
with tab1:
    st.header("Data Collection Phase")
    if technical_data is not None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          vertical_spacing=0.05,
                          row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=technical_data['time'],
                                    open=technical_data['Open'],
                                    high=technical_data['High'],
                                    low=technical_data['Low'],
                                    close=technical_data['Close'],
                                    name="OHLC"), row=1, col=1)
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
with tab2:
    st.header("Data Processing Phase")
    if technical_data is not None:
        technical_data['Price_Change'] = technical_data['Close'].pct_change()
        technical_data['Price_Change_Pct'] = technical_data['Price_Change'] * 100

        st.markdown("""
        ### Data Processing Methodology

        Data is processed through multiple steps to generate technical indicators and trend analysis:

        #### 1. Price Data Processing
        - **Input Data**: OHLC (Open, High, Low, Close) and Volume
        - **Data Cleaning**: Remove null and abnormal values
        - **Normalization**: Adjust prices for corporate events (if any)

        #### 2. Trend Indicators
        - **SMA (Simple Moving Average)**:
          - SMA20: 20-period moving average
          - SMA50: 50-period moving average
          - *Formula*: SMA = (P₁ + P₂ + ... + Pₙ) / n

        - **Bollinger Bands**:
          - Middle Band (BB_Mid): 20-period SMA
          - Upper Band (BB_High): BB_Mid + 2 × standard deviation
          - Lower Band (BB_Low): BB_Mid - 2 × standard deviation

        #### 3. Momentum Indicators
        - **RSI (Relative Strength Index)**:
          - Period: 14
          - Overbought: > 70
          - Oversold: < 30
          - *Formula*: RSI = 100 - (100 / (1 + RS))
            - RS = Average Gain / Average Loss

        - **MACD (Moving Average Convergence Divergence)**:
          - MACD Line: EMA(12) - EMA(26)
          - Signal Line: 9-period EMA of MACD
          - *Histogram*: MACD - Signal

        #### 4. Volatility Indicators
        - **ATR (Average True Range)**:
          - Period: 14
          - *True Range* = max(H-L, |H-C_prev|, |L-C_prev|)

        - **OBV (On Balance Volume)**:
          - Accumulates volume based on price direction
          - Increases when closing price rises
          - Decreases when closing price falls
        """)

        st.markdown("### Technical Analysis Charts")
        
        tech_tab1, tech_tab2, tech_tab3 = st.tabs([
            "Trend & Bollinger Bands",
            "Momentum Indicators",
            "Volatility & Volume"
        ])
        
        with tech_tab1:
            st.markdown("""
            #### Trend and Bollinger Bands Analysis
            
            - **Moving Average (MA)**: 
              - MA20 (short-term) and MA50 (medium-term) help identify trends
              - Crossover Up: Bullish signal
              - Crossover Down: Bearish signal
            
            - **Bollinger Bands**:
              - Band width shows volatility
              - Price touches upper band: Potential overbought
              - Price touches lower band: Potential oversold
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
                                        name="Price"), row=1, col=1)
            
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
            
            fig.update_layout(height=800, title="Trend and Bollinger Bands Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with tech_tab2:
            st.markdown("""
            #### Momentum Analysis
            
            - **RSI (Relative Strength Index)**:
              - Measures speed and magnitude of price movements
              - RSI > 70: Overbought zone
              - RSI < 30: Oversold zone
            
            - **MACD**:
              - MACD Line crosses Signal Line up: Buy signal
              - MACD Line crosses Signal Line down: Sell signal
              - Histogram shows trend strength
            
            - **Stochastic**:
              - %K and %D show price position within trading range
              - Overbought/oversold zones similar to RSI
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

            fig.update_layout(height=800, title="Momentum Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with tech_tab3:
            st.markdown("""
            #### Volatility and Volume Analysis
            
            - **ATR (Average True Range)**:
              - Measures average price volatility
              - High ATR: Strong volatility
              - Low ATR: Weak volatility
            
            - **Volume**:
              - Volume increases with price rise: Confirms uptrend
              - Volume increases with price fall: Confirms downtrend
              - Low volume: Lack of market interest
            
            - **OBV (On Balance Volume)**:
              - Accumulates volume based on price direction
              - OBV rises with price: Strong uptrend
              - OBV falls with rising price: Weak trend warning
            """)
            
            fig = make_subplots(rows=3, cols=1,
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.4, 0.3, 0.3],
                              subplot_titles=("ATR", "Volume", "OBV"))

            # ATR
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['ATR'],
                                   name="ATR",
                                   line=dict(color='blue')), row=1, col=1)

            colors = ['green' if x >= 0 else 'red' for x in technical_data['Price_Change']]
            fig.add_trace(go.Bar(x=technical_data['time'],
                               y=technical_data['Volume'],
                               name="Volume",
                               marker_color=colors), row=2, col=1)

            # OBV
            fig.add_trace(go.Scatter(x=technical_data['time'],
                                   y=technical_data['OBV'],
                                   name="OBV",
                                   line=dict(color='purple')), row=3, col=1)

            fig.update_layout(height=800, title="Volatility and Volume Analysis")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Trading Signals
        
        Based on the combination of indicators, the system identifies the following trading signals:
        
        #### Buy Signals:
        1. RSI < 30 (oversold)
        2. Price touches/breaks below Bollinger lower band
        3. MACD crosses above Signal Line
        4. Stochastic %K crosses above %D in oversold zone
        
        #### Sell Signals:
        1. RSI > 70 (overbought)
        2. Price touches/breaks above Bollinger upper band
        3. MACD crosses below Signal Line
        4. Stochastic %K crosses below %D in overbought zone
        
        #### Trend Confirmation:
        - **Uptrend** is confirmed when:
          - Price above MA20 and MA50
          - OBV increases with price
          - Volume increases on up days
        
        - **Downtrend** is confirmed when:
          - Price below MA20 and MA50
          - OBV decreases with price
          - Volume increases on down days
        """)

with tab3:
    st.header("Data Analysis Phase")
    
    if technical_data is not None:
        latest_data = technical_data.iloc[-1]
        
        technical_data['Price_Change'] = technical_data['Close'].pct_change()
        technical_data['Price_Change_Pct'] = technical_data['Price_Change'] * 100
        
        latest_rsi_signal = 'Overbought' if latest_data['RSI'] > 70 else 'Oversold' if latest_data['RSI'] < 30 else 'Neutral'
        latest_bb_position = 'Above upper band' if latest_data['Close'] > latest_data['BB_High'] else 'Below lower band' if latest_data['Close'] < latest_data['BB_Low'] else 'Inside bands'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend = "Up" if latest_data['SMA_20'] > latest_data['SMA_50'] else "Down"
            trend_color = "green" if trend == "Up" else "red"
            st.markdown(f"""
            <div class="stock-metrics">
                <h3>Price Trend</h3>
                <p>SMA20: {latest_data['SMA_20']:,.2f}</p>
                <p>SMA50: {latest_data['SMA_50']:,.2f}</p>
                <p>Trend: <span style="color: {trend_color}; font-weight: bold;">{trend}</span></p>
                <p>Volatility: {latest_data['ATR']:,.2f} ({(latest_data['ATR']/latest_data['Close']*100):,.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rsi_color = "red" if latest_data['RSI'] > 70 else "green" if latest_data['RSI'] < 30 else "black"
            st.markdown(f"""
            <div class="stock-metrics">
                <h3>Momentum Indicators</h3>
                <p>RSI: <span style="color: {rsi_color}; font-weight: bold;">{latest_data['RSI']:.2f}</span></p>
                <p>MACD: {latest_data['MACD']:.2f}</p>
                <p>MACD Signal: {latest_data['MACD_Signal']:.2f}</p>
                <p>Stochastic: {latest_data['Stoch']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            bb_color = "red" if latest_bb_position == 'Above upper band' else "green" if latest_bb_position == 'Below lower band' else "black"
            st.markdown(f"""
            <div class="stock-metrics">
                <h3>Bollinger Bands Analysis</h3>
                <p>Position: <span style="color: {bb_color}; font-weight: bold;">{latest_bb_position}</span></p>
                <p>BB Width: {(latest_data['BB_High'] - latest_data['BB_Low'])/latest_data['BB_Mid']*100:.2f}%</p>
                <p>Price Distance: {abs(latest_data['Close'] - latest_data['BB_Mid'])/latest_data['BB_Mid']*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Price Trend and Volume Analysis")
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25],
                           subplot_titles=("Price & Trend", "Volume", "Momentum"))

        fig.add_trace(go.Candlestick(x=technical_data['time'],
                                    open=technical_data['Open'],
                                    high=technical_data['High'],
                                    low=technical_data['Low'],
                                    close=technical_data['Close'],
                                    name="Price"), row=1, col=1)
        
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

        colors = ['green' if x >= 0 else 'red' for x in technical_data['Price_Change']]
        fig.add_trace(go.Bar(x=technical_data['time'],
                            y=technical_data['Volume'],
                            name="Volume",
                            marker_color=colors), row=2, col=1)

        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['RSI'],
                                name="RSI",
                                line=dict(color='purple')), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800,
                         showlegend=True,
                         title_text=f"Detailed Technical Analysis - {selected_stock}")
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Time-based Trend Analysis")
        
        periods = [5, 10, 20, 50]
        trend_data = {}
        
        for period in periods:
            changes = technical_data['Close'].pct_change(period) * 100
            if not changes.empty:
                trend_data[f'{period}-period Change'] = changes.iloc[-1]
            else:
                trend_data[f'{period}-period Change'] = 0
            
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            
            for period in periods:
                changes = technical_data['Close'].pct_change(period) * 100
                if not changes.empty:
                    fig.add_trace(go.Scatter(x=technical_data['time'],
                                           y=changes,
                                           name=f'{period}-period Change',
                                           line=dict(width=1)))
            
            fig.update_layout(title=f"Time-based Trend - {selected_stock}",
                            yaxis_title="Change (%)",
                            height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            trend_summary = pd.DataFrame({
                'Indicator': [
                    'Short-term Trend (5P)',
                    'Medium-term Trend (20P)',
                    'Long-term Trend (50P)',
                    'Volatility (ATR)',
                    'Momentum (RSI)',
                    'MACD Trend'
                ],
                'Status': [
                    'Up' if trend_data['5-period Change'] > 0 else 'Down',
                    'Up' if trend_data['20-period Change'] > 0 else 'Down',
                    'Up' if trend_data['50-period Change'] > 0 else 'Down',
                    f"{latest_data['ATR']:.2f} ({(latest_data['ATR']/latest_data['Close']*100):.2f}%)",
                    latest_rsi_signal,
                    'Up' if latest_data['MACD'] > latest_data['MACD_Signal'] else 'Down'
                ],
                'Value': [
                    f"{trend_data['5-period Change']:.2f}%",
                    f"{trend_data['20-period Change']:.2f}%",
                    f"{trend_data['50-period Change']:.2f}%",
                    f"{latest_data['Close']:.2f}",
                    f"{latest_data['RSI']:.2f}",
                    f"{latest_data['MACD']:.2f}"
                ]
            })
            
            st.dataframe(trend_summary,
                        column_config={
                            "Indicator": st.column_config.TextColumn("Indicator"),
                            "Status": st.column_config.TextColumn("Status"),
                            "Value": st.column_config.TextColumn("Value")
                        },
                        hide_index=True)

with tab4:
    st.header("Comprehensive Analysis")
    
    if technical_data is not None and trading_stats is not None:
        st.subheader("Trading Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stock-metrics">
                <h3>Price Statistics</h3>
                <p>52-Week High: {}</p>
                <p>52-Week Low: {}</p>
                <p>% Change from Low: {:.2f}%</p>
            </div>
            """.format(
                format_price(trading_stats.iloc[0]['high_price_1y']),
                format_price(trading_stats.iloc[0]['low_price_1y']),
                trading_stats.iloc[0]['pct_low_change_1y'] * 100
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stock-metrics">
                <h3>Foreign Statistics</h3>
                <p>Foreign Room: {:,.0f}</p>
                <p>Ownership: {:.2f}%</p>
                <p>Foreign Volume: {:,.0f}</p>
            </div>
            """.format(
                trading_stats.iloc[0]['foreign_room'],
                trading_stats.iloc[0]['current_holding_ratio'] * 100,
                trading_stats.iloc[0]['foreign_volume']
            ), unsafe_allow_html=True)
        
        st.subheader("Technical Analysis Summary")
        
        signals = pd.DataFrame(index=technical_data.index)
        signals['Signal'] = 'Hold'
        
        signals.loc[technical_data['RSI'] > 70, 'Signal'] = 'Sell'
        signals.loc[technical_data['RSI'] < 30, 'Signal'] = 'Buy'
        
        signals.loc[(technical_data['MACD'] > technical_data['MACD_Signal']) & 
                   (technical_data['MACD'].shift(1) <= technical_data['MACD_Signal'].shift(1)), 'Signal'] = 'Buy'
        signals.loc[(technical_data['MACD'] < technical_data['MACD_Signal']) & 
                   (technical_data['MACD'].shift(1) >= technical_data['MACD_Signal'].shift(1)), 'Signal'] = 'Sell'
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(x=technical_data['time'],
                                    open=technical_data['Open'],
                                    high=technical_data['High'],
                                    low=technical_data['Low'],
                                    close=technical_data['Close'],
                                    name="OHLC"))
        
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['SMA_20'],
                                name="SMA20",
                                line=dict(color='blue')))
        
        fig.add_trace(go.Scatter(x=technical_data['time'],
                                y=technical_data['SMA_50'],
                                name="SMA50",
                                line=dict(color='orange')))
        
        buy_signals = signals[signals['Signal'] == 'Buy']
        sell_signals = signals[signals['Signal'] == 'Sell']
        
        fig.add_trace(go.Scatter(
            x=technical_data.loc[buy_signals.index, 'time'],
            y=technical_data.loc[buy_signals.index, 'Low'] * 0.99,
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='green'),
            name='Buy Signal'
        ))
        
        fig.add_trace(go.Scatter(
            x=technical_data.loc[sell_signals.index, 'time'],
            y=technical_data.loc[sell_signals.index, 'High'] * 1.01,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Sell Signal'
        ))
        
        fig.update_layout(
            title=f"{selected_stock} - Technical Analysis Summary",
            yaxis_title="Price",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Company Information & News")
    
    def load_company_data(symbol):
        try:
            data = {}
            
            overview_path = COLLECTED_DATA_DIR / "company" / f"{symbol}_overview.csv"
            if overview_path.exists():
                data['overview'] = pd.read_csv(overview_path)
            else:
                st.error(f"Company overview file not found: {overview_path}")
                return None
            
            shareholders_path = COLLECTED_DATA_DIR / "company" / f"{symbol}_shareholders.csv"
            if shareholders_path.exists():
                data['shareholders'] = pd.read_csv(shareholders_path)
            else:
                st.warning(f"Shareholders information file not found: {shareholders_path}")
                data['shareholders'] = pd.DataFrame()
            
            officers_path = COLLECTED_DATA_DIR / "company" / f"{symbol}_officers_working.csv"
            if officers_path.exists():
                data['officers'] = pd.read_csv(officers_path)
            else:
                st.warning(f"Officers information file not found: {officers_path}")
                data['officers'] = pd.DataFrame()
            
            return data
            
        except Exception as e:
            st.error(f"Error reading company data: {str(e)}")
            return None

    def load_news_data(symbol):
        try:
            data = {}
            
            news_path = COLLECTED_DATA_DIR / "news" / f"{symbol}_news.csv"
            if news_path.exists():
                data['news'] = pd.read_csv(news_path)
            else:
                st.warning(f"News file not found: {news_path}")
                data['news'] = pd.DataFrame()
            
            events_path = COLLECTED_DATA_DIR / "news" / f"{symbol}_events.csv"
            if events_path.exists():
                data['events'] = pd.read_csv(events_path)
            else:
                st.warning(f"Events file not found: {events_path}")
                data['events'] = pd.DataFrame()
            
            return data
            
        except Exception as e:
            st.error(f"Error reading news data: {str(e)}")
            return None

    company_data = load_company_data(selected_stock)
    news_data = load_news_data(selected_stock)

    if company_data and 'overview' in company_data:
        st.subheader("Thông Tin Tổng Quan")
        overview = company_data['overview'].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ### {selected_stock} - {overview['icb_name3']}
            
            **Development History:**
            {overview['history']}
            
            **Company Profile:**
            {overview['company_profile']}
            """)
        
        with col2:
            st.markdown(f"""
            ### Basic Information
            - **Charter Capital:** {int(overview['charter_capital']):,} VND
            - **Outstanding Shares:** {int(overview['issue_share']):,}
            - **Industry:** {overview['icb_name3']}
            """)

        if 'shareholders' in company_data and not company_data['shareholders'].empty:
            st.subheader("Shareholder Structure")
            shareholders = company_data['shareholders']
            
            fig = px.pie(shareholders, 
                        values='share_own_percent', 
                        names='share_holder',
                        title=f"{selected_stock} Shareholder Structure")
            st.plotly_chart(fig, use_container_width=True)

        if 'officers' in company_data and not company_data['officers'].empty:
            st.subheader("Board of Directors")
            officers = company_data['officers']
            st.dataframe(
                officers[['officer_name', 'officer_position', 'quantity']],
                column_config={
                    "officer_name": "Name",
                    "officer_position": "Position",
                    "quantity": "Shares Owned"
                },
                hide_index=True
            )

    if news_data:
        st.subheader("News & Events")
        
        timeline_data = []
        
        if 'news' in news_data and not news_data['news'].empty:
            news = news_data['news']
            for _, row in news.iterrows():
                timeline_data.append({
                    'Date': pd.to_datetime(row['public_date']),
                    'Content': row['news_title'],
                    'Type': 'News',
                    'Link': row['news_source_link']
                })
        
        if 'events' in news_data and not news_data['events'].empty:
            events = news_data['events']
            for _, row in events.iterrows():
                timeline_data.append({
                    'Date': pd.to_datetime(row['public_date']),
                    'Content': row['event_title'],
                    'Type': 'Event',
                    'Link': row['source_url'] if pd.notna(row['source_url']) else ''
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('Date', ascending=False)
            
            for _, row in timeline_df.head(10).iterrows():
                with st.expander(f"{row['Date'].strftime('%d/%m/%Y')} - {row['Content']}", expanded=False):
                    st.markdown(f"""
                    **Type:** {row['Type']}
                    
                    **Link:** [{row['Content']}]({row['Link']})
                    """)
    else:
        st.warning("Unable to load news data. Please check the data path and files.")