import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from textblob import TextBlob
import re

class EnhancedStockPredictor:
    def __init__(self, collected_data_dir: str = None, processed_data_dir: str = None):
        """Khởi tạo predictor với đường dẫn đến dữ liệu"""
        # Thiết lập logging
        self.setup_logging()
        
        # Thiết lập đường dẫn
        if collected_data_dir is None or processed_data_dir is None:
            current_dir = Path(__file__).parent.parent
            self.collected_dir = current_dir / "collected_data"
            self.processed_dir = current_dir / "processed_data"
        else:
            self.collected_dir = Path(collected_data_dir)
            self.processed_dir = Path(processed_data_dir)
            
        # Khởi tạo scaler và model params
        self.scaler = StandardScaler()
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device': 'cpu',
            'num_threads': 4,
            'max_bin': 63,
            'num_iterations': 100
        }

    def setup_logging(self):
        """Thiết lập logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_trading_stats(self, symbol: str) -> pd.DataFrame:
        """Load dữ liệu thống kê giao dịch"""
        try:
            stats_file = list(self.collected_dir.glob(f"trading_stats/{symbol}_trading_stats*.csv"))[-1]
            return pd.read_csv(stats_file)
        except Exception as e:
            logging.warning(f"Không thể load trading stats cho {symbol}: {str(e)}")
            return pd.DataFrame()

    def load_fundamental_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Load dữ liệu cơ bản"""
        fundamental_data = {}
        try:
            # Load các báo cáo tài chính
            for report_type in ['balance', 'income', 'cashflow', 'ratios']:
                file_path = self.collected_dir / f"fundamental/{symbol}_{report_type}.csv"
                if file_path.exists():
                    fundamental_data[report_type] = pd.read_csv(file_path)
            return fundamental_data
        except Exception as e:
            logging.warning(f"Không thể load fundamental data cho {symbol}: {str(e)}")
            return {}

    def calculate_sentiment_score(self, text: str) -> float:
        """Tính điểm sentiment từ text"""
        try:
            if pd.isna(text):
                return 0.0
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    def load_news_data(self, symbol: str) -> pd.DataFrame:
        """Load và phân tích dữ liệu tin tức"""
        try:
            news_data = []
            
            # Load tin tức từ các nguồn khác nhau
            for news_type in ['news', 'events', 'reports']:
                file_path = self.collected_dir / f"news/{symbol}_{news_type}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    if 'news_title' in df.columns:
                        df['sentiment'] = df['news_title'].apply(self.calculate_sentiment_score)
                        news_data.append(df)
            
            if news_data:
                combined_news = pd.concat(news_data, ignore_index=True)
                # Chỉ lấy các tin tức có sentiment
                combined_news = combined_news.dropna(subset=['sentiment'])
                if len(combined_news) > 0:
                    return combined_news
            return pd.DataFrame({'sentiment': [0.0]})  # Trả về neutral sentiment nếu không có tin tức
        except Exception as e:
            logging.warning(f"Không thể load news data cho {symbol}: {str(e)}")
            return pd.DataFrame({'sentiment': [0.0]})  # Trả về neutral sentiment trong trường hợp lỗi

    def calculate_fundamental_features(self, fundamental_data: Dict[str, pd.DataFrame]) -> Dict:
        """Tính toán các chỉ số cơ bản"""
        features = {}
        try:
            if 'ratios' in fundamental_data:
                ratios = fundamental_data['ratios'].iloc[-1]
                features.update({
                    'pe_ratio': ratios.get('P/E', np.nan),
                    'pb_ratio': ratios.get('P/B', np.nan),
                    'roe': ratios.get('ROE', np.nan),
                    'roa': ratios.get('ROA', np.nan),
                    'current_ratio': ratios.get('Current Ratio', np.nan)
                })
            
            if 'income' in fundamental_data:
                income = fundamental_data['income']
                features['revenue_growth'] = (
                    income['Revenue (Bn. VND)'].iloc[-1] / income['Revenue (Bn. VND)'].iloc[-2] - 1
                    if len(income) >= 2 else np.nan
                )
                features['profit_growth'] = (
                    income['Attribute to parent company (Bn. VND)'].iloc[-1] / income['Attribute to parent company (Bn. VND)'].iloc[-2] - 1
                    if len(income) >= 2 else np.nan
                )
            
            return features
        except Exception as e:
            logging.warning(f"Lỗi khi tính fundamental features: {str(e)}")
            return {}

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo kỹ thuật"""
        # Thêm % thay đổi giá và khối lượng
        df['price_pct_change'] = df['close'].pct_change()
        df['volume_pct_change'] = df['volume'].pct_change()
        
        # Tính SMA với các khoảng thời gian khác nhau
        for period in [5, 10, 20]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
            df[f'Volume_SMA_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Tính RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Tính MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Tính Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Tính momentum và volatility
        df['Momentum'] = df['close'] - df['close'].shift(10)
        df['Volatility'] = df['close'].rolling(window=10).std()
        
        return df

    def prepare_features(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Chuẩn bị features và target cho mô hình"""
        try:
            # Đảm bảo dữ liệu được sắp xếp theo thời gian
            df = df.sort_values('time').reset_index(drop=True)
            
            # Target: 1 nếu giá tăng trong phiên tiếp theo
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            
            # Load dữ liệu bổ sung
            trading_stats = self.load_trading_stats(symbol)
            fundamental_data = self.load_fundamental_data(symbol)
            news_data = self.load_news_data(symbol)
            
            # Tính toán features cơ bản
            fundamental_features = self.calculate_fundamental_features(fundamental_data)
            
            # Tạo features
            feature_sets = {
                'price_momentum': [
                    'price_pct_change',
                    'Momentum',
                    'Volatility'
                ],
                'moving_averages': [
                    'SMA_5', 'SMA_10', 'SMA_20',
                    'Volume_SMA_5', 'Volume_SMA_10', 'Volume_SMA_20'
                ],
                'technical': [
                    'RSI',
                    'MACD', 'MACD_Hist',
                    'BB_width'
                ],
                'volume': [
                    'volume_pct_change'
                ]
            }
            
            # Thêm các tín hiệu kỹ thuật
            df['trend_signal'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
            df['macd_signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, 0)
            df['bb_signal'] = np.where(
                df['close'] > df['BB_upper'], -1,
                np.where(df['close'] < df['BB_lower'], 1, 0)
            )
            df['vol_signal'] = np.where(
                df['volume'] > df['Volume_SMA_20'], 1, 0
            )
            
            # Thêm các tín hiệu vào features
            feature_sets['signals'] = ['trend_signal', 'macd_signal', 'bb_signal', 'vol_signal']
            
            # Thêm fundamental features
            if fundamental_features:
                for key, value in fundamental_features.items():
                    df[f'fundamental_{key}'] = value
                feature_sets['fundamental'] = [f'fundamental_{k}' for k in fundamental_features.keys()]
            
            # Thêm sentiment features từ news
            if not news_data.empty:
                df['news_sentiment'] = news_data['sentiment'].mean()
                feature_sets['sentiment'] = ['news_sentiment']
            
            # Kết hợp tất cả features
            features = []
            for feature_group in feature_sets.values():
                features.extend(feature_group)
            
            # Xóa các dòng có NaN trong features quan trọng
            important_features = [
                'price_pct_change', 'Momentum', 'Volatility',
                'RSI', 'MACD', 'BB_width', 'volume_pct_change'
            ]
            df = df.dropna(subset=important_features + ['target'])
            
            # Điền giá trị NaN còn lại bằng 0
            df = df.fillna(0)
            
            # Loại bỏ các dòng đầu do không đủ dữ liệu cho các chỉ báo
            df = df.iloc[26:]  # 26 dòng đầu không có đủ dữ liệu cho MACD
            
            if len(df) == 0:
                raise ValueError("Không đủ dữ liệu sau khi xử lý")
                
            X = df[features]
            y = df['target']
            
            return X, y
            
        except Exception as e:
            logging.error(f"Lỗi khi chuẩn bị features: {str(e)}")
            raise e

    def train_model(self, symbol: str) -> Dict:
        """Huấn luyện mô hình cho một mã cổ phiếu"""
        try:
            # Load và chuẩn bị dữ liệu
            df = pd.read_csv(self.collected_dir / f"daily/{symbol}_daily.csv")
            df = self.calculate_technical_indicators(df)
            X, y = self.prepare_features(df, symbol)
            
            if len(X) < 100:  # Kiểm tra đủ dữ liệu để train
                raise ValueError(f"Không đủ dữ liệu để train mô hình cho {symbol}")
            
            # Chia tập train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Chuẩn hóa dữ liệu
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Tạo LightGBM datasets
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            
            # Thiết lập callbacks
            callbacks = [
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(period=10)
            ]
            
            # Huấn luyện mô hình
            model = lgb.train(
                params=self.model_params,
                train_set=train_data,
                valid_sets=[valid_data],
                callbacks=callbacks
            )
            
            # Đánh giá mô hình
            y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Lưu mô hình và scaler
            model_dir = self.processed_dir / "models"
            model_dir.mkdir(exist_ok=True)
            
            model.save_model(str(model_dir / f"{symbol}_model.txt"))
            joblib.dump(self.scaler, model_dir / f"{symbol}_scaler.joblib")
            
            # Lưu feature names để dùng cho predict
            feature_names = pd.DataFrame({'feature_name': X.columns})
            feature_names.to_csv(model_dir / f"{symbol}_feature_names.csv", index=False)
            
            # Lưu feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importance()
            }).sort_values('importance', ascending=False)
            
            importance.to_csv(model_dir / f"{symbol}_feature_importance.csv", index=False)
            
            return {
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'n_features': len(X.columns),
                'best_iteration': model.best_iteration,
                'feature_importance': importance.to_dict('records')[:5]  # Top 5 features
            }
            
        except Exception as e:
            logging.error(f"Lỗi khi huấn luyện mô hình cho {symbol}: {str(e)}")
            return {}

    def predict_signals(self, symbol: str, lookback_days: int = 30) -> Dict:
        """Dự đoán tín hiệu giao dịch cho một mã"""
        try:
            # Load mô hình và scaler
            model_dir = self.processed_dir / "models"
            model = lgb.Booster(model_file=str(model_dir / f"{symbol}_model.txt"))
            scaler = joblib.load(model_dir / f"{symbol}_scaler.joblib")
            
            # Load feature names
            feature_names = pd.read_csv(model_dir / f"{symbol}_feature_names.csv")
            
            # Load dữ liệu mới nhất
            df = pd.read_csv(self.collected_dir / f"daily/{symbol}_daily.csv")
            df = df.sort_values('time', ascending=True).reset_index(drop=True)
            
            # Lấy đủ dữ liệu để tính các chỉ báo
            min_lookback = max(lookback_days, 50)  # Cần ít nhất 50 ngày để tính các chỉ báo
            df = df.tail(min_lookback)
            
            # Tính toán chỉ báo và chuẩn bị features
            df = self.calculate_technical_indicators(df)
            
            # Load dữ liệu bổ sung
            trading_stats = self.load_trading_stats(symbol)
            fundamental_data = self.load_fundamental_data(symbol)
            news_data = self.load_news_data(symbol)
            
            # Tính toán features cơ bản
            fundamental_features = self.calculate_fundamental_features(fundamental_data)
            
            # Thêm các tín hiệu kỹ thuật
            df['trend_signal'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
            df['macd_signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, 0)
            df['bb_signal'] = np.where(
                df['close'] > df['BB_upper'], -1,
                np.where(df['close'] < df['BB_lower'], 1, 0)
            )
            df['vol_signal'] = np.where(
                df['volume'] > df['Volume_SMA_20'], 1, 0
            )
            
            # Thêm fundamental features
            if fundamental_features:
                for key, value in fundamental_features.items():
                    df[f'fundamental_{key}'] = value
            
            # Thêm sentiment features từ news
            if not news_data.empty:
                df['news_sentiment'] = news_data['sentiment'].mean()
            
            # Điền giá trị NaN bằng 0
            df = df.fillna(0)
            
            # Lấy dòng cuối cùng để dự đoán
            last_row = df.iloc[-1:]
            
            # Đảm bảo có đủ các features cần thiết
            for feature in feature_names['feature_name']:
                if feature not in last_row.columns:
                    last_row[feature] = 0
            
            # Chọn features theo thứ tự đúng
            X = last_row[feature_names['feature_name']]
            
            # Chuẩn hóa và dự đoán
            X_scaled = scaler.transform(X)
            probabilities = model.predict(X_scaled)
            predictions = (probabilities > 0.5).astype(int)
            
            # Tạo tín hiệu kết hợp
            latest_signals = {
                'prediction': int(predictions[-1]),
                'probability': float(probabilities[-1]),
                'technical_signals': {
                    'trend_signal': int(last_row['trend_signal'].iloc[0]),
                    'macd_signal': int(last_row['macd_signal'].iloc[0]),
                    'bb_signal': int(last_row['bb_signal'].iloc[0]),
                    'vol_signal': int(last_row['vol_signal'].iloc[0]),
                    'rsi': float(last_row['RSI'].iloc[0])
                },
                'fundamental_signals': fundamental_features,
                'sentiment': float(news_data['sentiment'].mean())
            }
            
            return latest_signals
            
        except Exception as e:
            logging.error(f"Lỗi khi dự đoán tín hiệu cho {symbol}: {str(e)}")
            return {}

    def generate_trading_strategy(self, symbol: str) -> Dict:
        """Tạo chiến lược giao dịch dựa trên các tín hiệu"""
        signals = self.predict_signals(symbol)
        if not signals:
            return {}
            
        # Xác định độ mạnh của tín hiệu
        signal_strength = 0
        tech_signals = signals['technical_signals']
        
        # Cộng điểm cho các tín hiệu kỹ thuật
        if signals['prediction'] == 1:
            signal_strength += 1
        if tech_signals['trend_signal'] == 1:
            signal_strength += 1
        if tech_signals['macd_signal'] == 1:
            signal_strength += 1
        if tech_signals['bb_signal'] == -1:  # Oversold
            signal_strength += 1
        if 30 <= tech_signals['rsi'] <= 70:
            signal_strength += 1
            
        # Cộng điểm cho fundamental signals
        fund_signals = signals['fundamental_signals']
        if fund_signals.get('pe_ratio', float('inf')) < 15:  # P/E hợp lý
            signal_strength += 1
        if fund_signals.get('revenue_growth', 0) > 0.1:  # Tăng trưởng doanh thu > 10%
            signal_strength += 1
        if fund_signals.get('roe', 0) > 0.15:  # ROE > 15%
            signal_strength += 1
            
        # Cộng điểm cho sentiment
        if signals['sentiment'] > 0.2:  # Sentiment tích cực
            signal_strength += 1
            
        # Xác định hành động
        if signal_strength >= 7:
            action = "Strong Buy"
        elif signal_strength >= 5:
            action = "Buy"
        elif signal_strength >= 3:
            action = "Neutral"
        elif signal_strength >= 1:
            action = "Sell"
        else:
            action = "Strong Sell"
            
        strategy = {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': signals['probability'],
            'technical_analysis': {
                'trend_following': tech_signals['trend_signal'] == 1,
                'momentum': tech_signals['macd_signal'] == 1,
                'overbought_oversold': tech_signals['bb_signal'],
                'rsi_level': tech_signals['rsi']
            },
            'fundamental_analysis': fund_signals,
            'sentiment_analysis': {
                'score': signals['sentiment'],
                'interpretation': 'Positive' if signals['sentiment'] > 0 else 'Negative'
            }
        }
        
        return strategy

def main():
    # Khởi tạo predictor
    predictor = EnhancedStockPredictor()
    
    # Danh sách mã cần phân tích
    symbols = ['VCB', 'VNM', 'FPT']
    
    # Huấn luyện và đánh giá mô hình cho từng mã
    for symbol in symbols:
        logging.info(f"\nHuấn luyện mô hình cho {symbol}:")
        metrics = predictor.train_model(symbol)
        if metrics:
            logging.info(f"Accuracy: {metrics['accuracy']:.2f}")
            logging.info(f"Precision: {metrics['precision']:.2f}")
            logging.info(f"Recall: {metrics['recall']:.2f}")
            logging.info(f"F1-score: {metrics['f1']:.2f}")
            logging.info("\nTop 5 features quan trọng nhất:")
            for feature in metrics['feature_importance']:
                logging.info(f"- {feature['feature']}: {feature['importance']}")
            
        # Tạo chiến lược giao dịch
        strategy = predictor.generate_trading_strategy(symbol)
        if strategy:
            logging.info(f"\nChiến lược giao dịch cho {symbol}:")
            logging.info(f"Hành động: {strategy['action']}")
            logging.info(f"Độ tin cậy: {strategy['confidence']:.2f}")
            logging.info(f"Độ mạnh tín hiệu: {strategy['signal_strength']}/9")
            
            logging.info("\nPhân tích kỹ thuật:")
            logging.info(f"- Trend Following: {'Tích cực' if strategy['technical_analysis']['trend_following'] else 'Tiêu cực'}")
            logging.info(f"- Momentum: {'Tích cực' if strategy['technical_analysis']['momentum'] else 'Tiêu cực'}")
            logging.info(f"- RSI: {strategy['technical_analysis']['rsi_level']:.2f}")
            
            logging.info("\nPhân tích cơ bản:")
            fund = strategy['fundamental_analysis']
            logging.info(f"- P/E: {fund.get('pe_ratio', 'N/A')}")
            logging.info(f"- ROE: {fund.get('roe', 'N/A'):.2%}")
            logging.info(f"- Tăng trưởng doanh thu: {fund.get('revenue_growth', 'N/A'):.2%}")
            
            logging.info("\nPhân tích sentiment:")
            logging.info(f"- Điểm: {strategy['sentiment_analysis']['score']:.2f}")
            logging.info(f"- Đánh giá: {strategy['sentiment_analysis']['interpretation']}")
        
        logging.info("------------------------")

if __name__ == "__main__":
    main() 