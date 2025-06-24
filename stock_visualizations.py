import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import plotly.subplots as sp

# Sample data generation (replace this with your actual data)
def generate_sample_data():
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Cumulative returns data
    data = {
        'Date': dates,
        'VNM': np.random.normal(0.0002, 0.02, len(dates)).cumsum() + 0.25,
        'FPT': np.random.normal(0.0003, 0.02, len(dates)).cumsum() + 0.32,
        'VCB': np.random.normal(0.0001, 0.02, len(dates)).cumsum() + 0.20,
        'VN-Index': np.random.normal(0.0001, 0.015, len(dates)).cumsum() + 0.18
    }
    return pd.DataFrame(data)

# 1. Line Chart: Cumulative Returns
def plot_cumulative_returns():
    df = generate_sample_data()
    
    fig = go.Figure()
    
    # Add traces for each stock
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['VNM']*100,
        name='VNM',
        line=dict(color='#28A745')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['FPT']*100,
        name='FPT',
        line=dict(color='#007BFF')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['VCB']*100,
        name='VCB',
        line=dict(color='#DC3545')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['VN-Index']*100,
        name='VN-Index',
        line=dict(color='#6C757D')
    ))
    
    fig.update_layout(
        title='Lợi nhuận tích lũy của chiến lược lướt sóng (2020-2024)',
        xaxis_title='Thời gian',
        yaxis_title='Lợi nhuận tích lũy (%)',
        template='plotly_white',
        hovermode='x unified',
        width=1200,
        height=800
    )
    
    fig.write_image('cumulative_returns.png', scale=2)

# 2. Bar Chart: Win Rate and Sharpe Ratio
def plot_win_rate_sharpe():
    # Sample data
    stocks = ['VNM', 'FPT', 'VCB']
    win_rates = [65, 70, 60]
    sharpe_ratios = [1.0, 1.2, 0.9]
    
    # Tạo figure với hai trục y
    fig = go.Figure()
    
    # Thêm cột tỷ lệ thắng
    fig.add_trace(
        go.Bar(
            name='Tỷ lệ thắng (%)',
            x=stocks,
            y=win_rates,
            marker_color='#007BFF',
            yaxis='y'
        )
    )
    
    # Thêm cột Sharpe ratio
    fig.add_trace(
        go.Bar(
            name='Tỷ lệ Sharpe',
            x=stocks,
            y=sharpe_ratios,
            marker_color='#FFC107',
            yaxis='y2'
        )
    )
    
    # Cập nhật layout với hai trục y
    fig.update_layout(
        title='Tỷ lệ thắng và tỷ lệ Sharpe theo mã cổ phiếu',
        template='plotly_white',
        width=1000,
        height=600,
        yaxis=dict(
            title=dict(
                text='Tỷ lệ thắng (%)',
                font=dict(color='#007BFF')
            ),
            tickfont=dict(color='#007BFF'),
            range=[0, 100]  # Giới hạn từ 0-100%
        ),
        yaxis2=dict(
            title=dict(
                text='Tỷ lệ Sharpe',
                font=dict(color='#FFC107')
            ),
            tickfont=dict(color='#FFC107'),
            anchor='x',
            overlaying='y',
            side='right',
            range=[0, 2]  # Giới hạn từ 0-2 cho Sharpe ratio
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # Thêm lưới dọc để dễ đọc
        xaxis=dict(
            showgrid=False
        ),
        yaxis_gridcolor='rgba(0,123,255,0.1)',
        yaxis2_gridcolor='rgba(255,193,7,0.1)'
    )
    
    # Thêm annotations cho mỗi cột
    for i in range(len(stocks)):
        # Annotation cho tỷ lệ thắng
        fig.add_annotation(
            x=stocks[i],
            y=win_rates[i],
            text=f"{win_rates[i]}%",
            yanchor='bottom',
            showarrow=False,
            font=dict(color='#007BFF')
        )
        # Annotation cho Sharpe ratio
        fig.add_annotation(
            x=stocks[i],
            y=sharpe_ratios[i],
            text=f"{sharpe_ratios[i]:.1f}",
            yanchor='bottom',
            yshift=10,
            showarrow=False,
            font=dict(color='#FFC107')
        )
    
    fig.write_image('win_rate_sharpe.png', scale=2)

# 3. Heatmap: Technical Indicators Correlation
def plot_correlation_heatmap():
    # Sample correlation data
    indicators = ['Giá đóng cửa', 'RSI', 'MACD', 'MFI', 'Volume']
    
    # Sample correlation matrices for each stock
    correlations = {
        'FPT': np.array([
            [1.00, 0.65, 0.45, 0.60, 0.30],
            [0.65, 1.00, 0.40, 0.55, 0.25],
            [0.45, 0.40, 1.00, 0.35, 0.20],
            [0.60, 0.55, 0.35, 1.00, 0.30],
            [0.30, 0.25, 0.20, 0.30, 1.00]
        ]),
        'VNM': np.array([
            [1.00, 0.50, 0.40, 0.45, 0.25],
            [0.50, 1.00, 0.35, 0.40, 0.20],
            [0.40, 0.35, 1.00, 0.30, 0.15],
            [0.45, 0.40, 0.30, 1.00, 0.25],
            [0.25, 0.20, 0.15, 0.25, 1.00]
        ]),
        'VCB': np.array([
            [1.00, 0.45, 0.35, 0.40, 0.20],
            [0.45, 1.00, 0.30, 0.35, 0.15],
            [0.35, 0.30, 1.00, 0.25, 0.10],
            [0.40, 0.35, 0.25, 1.00, 0.20],
            [0.20, 0.15, 0.10, 0.20, 1.00]
        ])
    }
    
    # Create subplots for each stock
    fig = sp.make_subplots(rows=1, cols=3, 
                          subplot_titles=('FPT', 'VNM', 'VCB'),
                          horizontal_spacing=0.1)
    
    for idx, (stock, corr_matrix) in enumerate(correlations.items(), 1):
        heatmap = go.Heatmap(
            z=corr_matrix,
            x=indicators,
            y=indicators,
            colorscale=[[0, '#DC3545'], [0.5, '#FFFFFF'], [1, '#28A745']],
            zmin=-1,
            zmax=1,
            showscale=True if idx == 3 else False,
            text=[[f'{val:.2f}' for val in row] for row in corr_matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x}<br>%{y}<br>Tương quan: %{text}<extra></extra>'
        )
        fig.add_trace(heatmap, row=1, col=idx)
    
    fig.update_layout(
        title='Tương quan giữa các chỉ báo kỹ thuật và giá',
        height=600,
        width=1500,
        template='plotly_white'
    )
    
    # Cập nhật font size cho tiêu đề subplot
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=14, weight='bold'))
    
    fig.write_image('correlation_heatmap.png', scale=2)

def main():
    plot_cumulative_returns()
    plot_win_rate_sharpe()
    plot_correlation_heatmap()

if __name__ == "__main__":
    main() 