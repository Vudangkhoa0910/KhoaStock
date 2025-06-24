# PIPELINE PHƯƠNG PHÁP NGHIÊN CỨU KHOASTOCK

```mermaid
graph TD
    subgraph Data_Collection[3.2 Thu thập dữ liệu]
        A1[OHLCV Data] --> A2[Dữ liệu thị trường]
        A3[Financial Data] --> A2
        A4[News Data] --> A2
        A2 --> A5[Data Storage]
    end

    subgraph Data_Preprocessing[3.3 Tiền xử lý dữ liệu]
        B1[Xử lý missing values] --> B2[Chuẩn hóa dữ liệu]
        B2 --> B3[Feature Engineering]
        B3 --> B4[Data Validation]
    end

    subgraph Data_Analysis[3.4 Phân tích dữ liệu]
        C1[Technical Analysis] --> C4[Combined Analysis]
        C2[Statistical Analysis] --> C4
        C3[Sentiment Analysis] --> C4
    end

    subgraph Model_Building[3.5 Xây dựng mô hình]
        D1[LSTM Model] --> D3[Ensemble Model]
        D2[Traditional ML Models] --> D3
    end

    subgraph Evaluation[3.6 Dự đoán và đánh giá]
        E1[Performance Metrics] --> E3[Final Evaluation]
        E2[Risk Metrics] --> E3
    end

    Data_Collection --> Data_Preprocessing
    Data_Preprocessing --> Data_Analysis
    Data_Analysis --> Model_Building
    Model_Building --> Evaluation
```

## Chi tiết các bước

### 1. Thu thập dữ liệu (3.2)
```python
# Pipeline thu thập dữ liệu
data_collection_pipeline = {
    'market_data': {
        'source': 'vnstock API',
        'frequency': 'realtime',
        'data_types': ['OHLCV', 'Index', 'Derivatives']
    },
    'financial_data': {
        'source': 'company reports',
        'frequency': 'quarterly',
        'data_types': ['Balance Sheet', 'Income Statement', 'Cash Flow']
    },
    'news_data': {
        'source': 'news APIs',
        'frequency': 'continuous',
        'data_types': ['Market News', 'Company News']
    }
}
```

### 2. Tiền xử lý dữ liệu (3.3)
```python
# Pipeline tiền xử lý
preprocessing_pipeline = {
    'missing_values': {
        'numerical': 'interpolation',
        'categorical': 'mode filling',
        'time_series': 'forward fill'
    },
    'normalization': {
        'price_data': 'min-max scaling',
        'financial_ratios': 'z-score normalization',
        'technical_indicators': 'decimal scaling'
    },
    'feature_engineering': {
        'technical': ['SMA', 'EMA', 'RSI', 'MACD'],
        'fundamental': ['P/E', 'P/B', 'ROE', 'ROA'],
        'sentiment': ['News Score', 'Market Sentiment']
    }
}
```

### 3. Phân tích dữ liệu (3.4)
```python
# Pipeline phân tích
analysis_pipeline = {
    'technical_analysis': {
        'trend_analysis': ['Moving Averages', 'Trend Lines'],
        'momentum_analysis': ['RSI', 'MACD', 'Stochastic'],
        'volatility_analysis': ['Bollinger Bands', 'ATR']
    },
    'statistical_analysis': {
        'correlation': 'Pearson Correlation',
        'stationarity': 'ADF Test',
        'normality': 'Shapiro-Wilk Test'
    },
    'sentiment_analysis': {
        'news_processing': 'NLP',
        'sentiment_scoring': 'VADER',
        'market_sentiment': 'Aggregated Score'
    }
}
```

### 4. Xây dựng mô hình (3.5)
```python
# Pipeline mô hình
model_pipeline = {
    'deep_learning': {
        'architecture': 'LSTM',
        'layers': [
            {'type': 'LSTM', 'units': 50},
            {'type': 'Dropout', 'rate': 0.2},
            {'type': 'Dense', 'units': 1}
        ]
    },
    'ensemble': {
        'models': ['RandomForest', 'XGBoost', 'LightGBM'],
        'stacking': 'LinearRegression'
    }
}
```

### 5. Đánh giá hiệu quả (3.6)
```python
# Pipeline đánh giá
evaluation_pipeline = {
    'performance_metrics': {
        'accuracy': ['MAE', 'RMSE', 'Direction Accuracy'],
        'trading': ['Returns', 'Sharpe Ratio', 'Max Drawdown']
    },
    'validation': {
        'cross_validation': '5-fold',
        'backtesting': 'Walk-forward optimization'
    },
    'monitoring': {
        'model_drift': 'KS-test',
        'performance_decay': 'Moving window analysis'
    }
}
```

## Luồng dữ liệu

```mermaid
sequenceDiagram
    participant DC as Data Collection
    participant DP as Data Preprocessing
    participant DA as Data Analysis
    participant MB as Model Building
    participant EV as Evaluation

    DC->>DP: Raw Data
    DP->>DA: Processed Data
    DA->>MB: Analysis Results
    MB->>EV: Model Predictions
    EV->>DC: Feedback Loop

    loop Real-time Update
        DC->>EV: Continuous Data Flow
    end
```

## Kiến trúc hệ thống

```mermaid
graph LR
    subgraph Data Layer
        D1[Raw Data] --> D2[Processed Data]
        D2 --> D3[Feature Store]
    end

    subgraph Processing Layer
        P1[Data Pipeline] --> P2[Analysis Pipeline]
        P2 --> P3[Model Pipeline]
    end

    subgraph Service Layer
        S1[API Services] --> S2[Model Services]
        S2 --> S3[Monitoring Services]
    end

    Data Layer --> Processing Layer
    Processing Layer --> Service Layer
``` 