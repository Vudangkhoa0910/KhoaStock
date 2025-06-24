# 3. PHƯƠNG PHÁP NGHIÊN CỨU

## 3.1. Tổng quan về phương pháp luận

### Quy trình nghiên cứu
1. Thu thập dữ liệu đa nguồn
2. Tiền xử lý và làm sạch dữ liệu
3. Phân tích kỹ thuật và thống kê
4. Xây dựng các chỉ báo tổng hợp
5. Đánh giá và tối ưu hóa
6. Triển khai và giám sát

### Phương pháp tiếp cận
- Phân tích kỹ thuật (Technical Analysis)
- Phân tích cơ bản (Fundamental Analysis)
- Phân tích tâm lý thị trường (Market Sentiment)
- Phân tích thống kê đa biến

## 3.2. Thu thập dữ liệu

### Dữ liệu thị trường (OHLCV)
```python
# Cấu trúc dữ liệu OHLCV
market_data = {
    'open': daily_open_prices,      # Giá mở cửa
    'high': daily_high_prices,      # Giá cao nhất
    'low': daily_low_prices,        # Giá thấp nhất
    'close': daily_close_prices,    # Giá đóng cửa
    'volume': daily_trading_volumes  # Khối lượng giao dịch
}
```

### Chỉ số tài chính cơ bản
```python
# Financial Ratios
def calculate_financial_ratios(market_price, financial_data):
    ratios = {
        'P_E': market_price / financial_data['earnings_per_share'],
        'P_B': market_price / financial_data['book_value_per_share'],
        'ROE': (financial_data['net_income'] / financial_data['shareholders_equity']) * 100,
        'ROA': (financial_data['net_income'] / financial_data['total_assets']) * 100,
        'Current_ratio': financial_data['current_assets'] / financial_data['current_liabilities'],
        'Debt_equity': financial_data['total_liabilities'] / financial_data['shareholders_equity'],
        'Profit_margin': (financial_data['net_income'] / financial_data['revenue']) * 100
    }
    return ratios
```

## 3.3. Tiền xử lý dữ liệu

### Xử lý dữ liệu thiếu và nhiễu
```python
def process_missing_data(df):
    # Forward Fill cho dữ liệu chuỗi thời gian
    df_processed = df.fillna(method='ffill')
    
    # Interpolation cho dữ liệu số
    df_processed = df_processed.interpolate(method='linear')
    
    # Xử lý outliers bằng IQR
    def remove_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    numeric_columns = df_processed.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df_processed[col] = remove_outliers(df_processed[col])
    
    return df_processed
```

### Chuẩn hóa dữ liệu
```python
def normalize_data(df):
    # Min-Max Scaling cho dữ liệu giá
    def min_max_scale(series):
        return (series - series.min()) / (series.max() - series.min())
    
    # Z-score Normalization cho chỉ số tài chính
    def z_score_normalize(series):
        return (series - series.mean()) / series.std()
    
    # Log Transform cho volume
    def log_transform(series):
        return np.log1p(series)
    
    price_cols = ['open', 'high', 'low', 'close']
    ratio_cols = ['P_E', 'P_B', 'ROE', 'ROA']
    volume_cols = ['volume']
    
    for col in price_cols:
        df[f'{col}_normalized'] = min_max_scale(df[col])
    
    for col in ratio_cols:
        df[f'{col}_normalized'] = z_score_normalize(df[col])
    
    for col in volume_cols:
        df[f'{col}_normalized'] = log_transform(df[col])
    
    return df
```

## 3.4. Phân tích dữ liệu

### Chỉ báo kỹ thuật

#### Moving Averages và Trend
```python
def calculate_trend_indicators(df, periods=[20, 50, 200]):
    """Tính toán các chỉ báo xu hướng"""
    
    # Simple Moving Average (SMA)
    for period in periods:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Average (EMA)
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Directional Movement Index (DMI)
    def calculate_dmi(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return plus_di, minus_di, adx
    
    df['Plus_DI'], df['Minus_DI'], df['ADX'] = calculate_dmi(df['high'], df['low'], df['close'])
    
    return df
```

#### Momentum Indicators
```python
def calculate_momentum_indicators(df):
    """Tính toán các chỉ báo động lượng"""
    
    # Relative Strength Index (RSI)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = calculate_rsi(df['close'])
    
    # Stochastic Oscillator
    def calculate_stochastic(data, k_period=14, d_period=3):
        low_min = data['low'].rolling(k_period).min()
        high_max = data['high'].rolling(k_period).max()
        k = 100 * (data['close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k, d
    
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df)
    
    # Rate of Change (ROC)
    def calculate_roc(data, period=12):
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    df['ROC'] = calculate_roc(df['close'])
    
    # Money Flow Index (MFI)
    def calculate_mfi(high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    df['MFI'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
    
    return df
```

#### Volatility Indicators
```python
def calculate_volatility_indicators(df):
    """Tính toán các chỉ báo biến động"""
    
    # Bollinger Bands
    def calculate_bollinger_bands(data, window=20, num_std=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['close'])
    
    # Average True Range (ATR)
    def calculate_atr(high, low, close, period=14):
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr.max(axis=1)
        return tr['tr'].rolling(period).mean()
    
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Keltner Channel
    def calculate_keltner_channel(high, low, close, period=20, atr_period=10, multiplier=2):
        basis = close.rolling(period).mean()
        atr = calculate_atr(high, low, close, atr_period)
        upper = basis + multiplier * atr
        lower = basis - multiplier * atr
        return upper, basis, lower
    
    df['KC_Upper'], df['KC_Middle'], df['KC_Lower'] = calculate_keltner_channel(
        df['high'], df['low'], df['close']
    )
    
    return df
```

### Phân tích thống kê

#### Correlation Analysis
```python
def perform_statistical_analysis(df):
    """Thực hiện phân tích thống kê"""
    
    # Pearson Correlation
    correlation_matrix = df[['close', 'volume', 'RSI', 'MACD']].corr()
    
    # Spearman Rank Correlation
    spearman_corr = df[['close', 'volume']].corr(method='spearman')
    
    # Autocorrelation
    def calculate_autocorrelation(series, lags=40):
        return [series.autocorr(lag=i) for i in range(1, lags+1)]
    
    price_autocorr = calculate_autocorrelation(df['close'])
    volume_autocorr = calculate_autocorrelation(df['volume'])
    
    # Granger Causality Test
    from statsmodels.tsa.stattools import grangercausalitytests
    
    def granger_causality(data1, data2, maxlag=5):
        data = pd.concat([data1, data2], axis=1)
        return grangercausalitytests(data, maxlag=maxlag, verbose=False)
    
    # Kiểm tra tính dừng
    from statsmodels.tsa.stattools import adfuller
    
    def check_stationarity(series):
        result = adfuller(series)
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
    
    analysis_results = {
        'correlation_matrix': correlation_matrix,
        'spearman_correlation': spearman_corr,
        'price_autocorrelation': price_autocorr,
        'volume_autocorrelation': volume_autocorr,
        'price_stationarity': check_stationarity(df['close']),
        'volume_stationarity': check_stationarity(df['volume'])
    }
    
    return analysis_results
```

## 3.5. Xây dựng chỉ báo tổng hợp

### Composite Technical Indicator
```python
def calculate_composite_indicator(df):
    """Tính toán chỉ báo tổng hợp từ nhiều chỉ báo kỹ thuật"""
    
    # Chuẩn hóa các chỉ báo
    def normalize_indicator(series):
        return (series - series.mean()) / series.std()
    
    # Trọng số cho từng chỉ báo
    weights = {
        'RSI': 0.2,
        'MACD': 0.2,
        'BB_%B': 0.15,
        'Stoch_K': 0.15,
        'ADX': 0.15,
        'MFI': 0.15
    }
    
    # Tính toán Bollinger %B
    df['BB_%B'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Chuẩn hóa các chỉ báo
    normalized_indicators = {}
    for indicator in weights.keys():
        normalized_indicators[indicator] = normalize_indicator(df[indicator])
    
    # Tính toán chỉ báo tổng hợp
    composite = sum(normalized_indicators[ind] * weight for ind, weight in weights.items())
    
    return composite
```

### Market Strength Index
```python
def calculate_market_strength(df):
    """Tính toán chỉ số sức mạnh thị trường"""
    
    # Volume Price Trend (VPT)
    df['VPT'] = df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    df['VPT_MA'] = df['VPT'].rolling(window=20).mean()
    
    # Accumulation/Distribution Line (ADL)
    df['Money_Flow_Multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['Money_Flow_Volume'] = df['Money_Flow_Multiplier'] * df['volume']
    df['ADL'] = df['Money_Flow_Volume'].cumsum()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
    
    # Chaikin Money Flow (CMF)
    period = 20
    df['CMF'] = df['Money_Flow_Volume'].rolling(period).sum() / df['volume'].rolling(period).sum()
    
    # Market Strength Index
    indicators = ['VPT', 'ADL', 'OBV', 'CMF']
    normalized_indicators = {}
    
    for indicator in indicators:
        normalized_indicators[indicator] = (df[indicator] - df[indicator].rolling(period).min()) / \
                                        (df[indicator].rolling(period).max() - df[indicator].rolling(period).min())
    
    df['Market_Strength'] = sum(normalized_indicators.values()) / len(indicators)
    
    return df
```

## 3.6. Đánh giá hiệu quả

### Performance Metrics
```python
def calculate_trading_metrics(prices, signals):
    """Tính toán các chỉ số hiệu quả giao dịch"""
    
    # Tính toán lợi nhuận
    returns = prices.pct_change()
    strategy_returns = returns * signals.shift(1)
    
    # Sharpe Ratio
    risk_free_rate = 0.02  # Lãi suất phi rủi ro
    excess_returns = strategy_returns - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Maximum Drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Win Rate
    winning_trades = strategy_returns[strategy_returns > 0].count()
    total_trades = strategy_returns[strategy_returns != 0].count()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit Factor
    gross_profits = strategy_returns[strategy_returns > 0].sum()
    gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    return {
        'Total_Return': (cum_returns.iloc[-1] - 1) * 100,
        'Annual_Return': ((cum_returns.iloc[-1]) ** (252/len(returns)) - 1) * 100,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown * 100,
        'Win_Rate': win_rate * 100,
        'Profit_Factor': profit_factor
    }
```

### Risk Analysis
```python
def analyze_risk(returns):
    """Phân tích các chỉ số rủi ro"""
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Conditional Value at Risk (CVaR/Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # Volatility
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Beta (relative to market)
    def calculate_beta(returns, market_returns):
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    
    # Sortino Ratio
    risk_free_rate = 0.02
    excess_returns = returns - risk_free_rate/252
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2))
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
    
    return {
        'VaR_95': var_95 * 100,
        'VaR_99': var_99 * 100,
        'CVaR_95': cvar_95 * 100,
        'CVaR_99': cvar_99 * 100,
        'Annual_Volatility': annual_volatility * 100,
        'Sortino_Ratio': sortino_ratio
    }
```

Phân tích kỹ thuật chi tiết:
Chỉ báo xu hướng (Trend): SMA, EMA, MACD, DMI
Chỉ báo động lượng (Momentum): RSI, Stochastic, ROC, MFI
Chỉ báo biến động (Volatility): Bollinger Bands, ATR, Keltner Channel
Phân tích thống kê nâng cao:
Tương quan Pearson và Spearman
Phân tích tự tương quan (Autocorrelation)
Kiểm định nhân quả Granger
Kiểm định tính dừng
Chỉ báo tổng hợp:
Composite Technical Indicator với trọng số
Market Strength Index dựa trên volume
Các chỉ số đo lường sức mạnh thị trường
Đánh giá hiệu quả giao dịch:
Các metrics về lợi nhuận và rủi ro
Phân tích drawdown và win rate
Đánh giá rủi ro đa chiều (VaR, CVaR, Beta)


## 3.7. Cân nhắc đạo đức

### Bảo mật dữ liệu
- Mã hóa dữ liệu nhạy cảm
- Kiểm soát quyền truy cập
- Ghi log hoạt động

### Minh bạch
- Công khai phương pháp phân tích
- Theo dõi và audit quyết định
- Cập nhật thông tin realtime


## 3.9. Tóm tắt

Phương pháp nghiên cứu của dự án KhoaStock:
1. Thu thập dữ liệu đa nguồn tự động
2. Tiền xử lý dữ liệu chuyên sâu
3. Phân tích kỹ thuật và thống kê
4. Xây dựng chỉ báo tổng hợp
5. Đánh giá đa chiều
6. Đảm bảo tính minh bạch và đạo đức