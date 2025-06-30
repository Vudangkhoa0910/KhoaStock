#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

class UltraLightStockPredictor:
    
    def __init__(self, symbols=['VCB', 'VNM', 'FPT']):
        self.symbols = symbols
        self.data = {}
        self.models = {}
        self.predictions = {}
        self.feature_stats = {}
        
    def load_data(self, data_path="/Users/vudangkhoa/Working/KhoaStock/data/collected_data"):
        """Tải dữ liệu cho tất cả các mã"""
        print("📊 Đang tải dữ liệu...")
        
        for symbol in self.symbols:
            try:
                # Đọc dữ liệu technical
                tech_file = f"{data_path}/technical/{symbol}_technical.csv"
                if os.path.exists(tech_file):
                    tech_data = pd.read_csv(tech_file)
                    tech_data['time'] = pd.to_datetime(tech_data['time'])
                    tech_data = tech_data.sort_values('time').reset_index(drop=True)
                    self.data[symbol] = tech_data
                    print(f"✅ Đã tải {len(tech_data)} dòng dữ liệu cho {symbol}")
                else:
                    print(f"❌ Không tìm thấy file {tech_file}")
                    
            except Exception as e:
                print(f"❌ Lỗi khi tải dữ liệu {symbol}: {str(e)}")
    
    def simple_moving_average(self, data, window):
        """Tính trung bình động đơn giản"""
        result = []
        for i in range(len(data)):
            if i < window - 1:
                result.append(np.nan)
            else:
                avg = sum(data[i-window+1:i+1]) / window
                result.append(avg)
        return result
    
    def calculate_rsi(self, prices, period=14):
        """Tính RSI đơn giản"""
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(delta, 0) for delta in deltas]
        losses = [abs(min(delta, 0)) for delta in deltas]
        
        rsi_values = [np.nan] * period
        
        if len(gains) >= period:
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    def create_features(self, df):
        """Tạo features đơn giản"""
        features_df = df.copy()
        
        # Giá đóng cửa
        close_prices = features_df['Close'].tolist()
        
        # Tính toán các features cơ bản
        features_df['price_change_1d'] = features_df['Close'].pct_change().fillna(0)
        features_df['price_change_3d'] = features_df['Close'].pct_change(3).fillna(0)
        features_df['price_change_5d'] = features_df['Close'].pct_change(5).fillna(0)
        
        # Volume change
        features_df['volume_change'] = features_df['Volume'].pct_change().fillna(0)
        
        # High-Low ratio
        features_df['hl_ratio'] = ((features_df['High'] - features_df['Low']) / features_df['Close']).fillna(0)
        
        # Open-Close ratio
        features_df['oc_ratio'] = ((features_df['Close'] - features_df['Open']) / features_df['Open']).fillna(0)
        
        # Moving averages
        features_df['ma5'] = self.simple_moving_average(close_prices, 5)
        features_df['ma10'] = self.simple_moving_average(close_prices, 10)
        features_df['ma20'] = self.simple_moving_average(close_prices, 20)
        
        # MA ratios
        features_df['price_ma5_ratio'] = (features_df['Close'] / features_df['ma5']).fillna(1)
        features_df['price_ma10_ratio'] = (features_df['Close'] / features_df['ma10']).fillna(1)
        features_df['price_ma20_ratio'] = (features_df['Close'] / features_df['ma20']).fillna(1)
        
        # RSI nếu chưa có
        if 'RSI' not in features_df.columns:
            features_df['RSI'] = self.calculate_rsi(close_prices)
        
        # Lag features
        features_df['close_lag1'] = features_df['Close'].shift(1).fillna(features_df['Close'])
        features_df['close_lag2'] = features_df['Close'].shift(2).fillna(features_df['Close'])
        features_df['volume_lag1'] = features_df['Volume'].shift(1).fillna(features_df['Volume'])
        
        # Target
        features_df['target_price_change'] = features_df['Close'].shift(-1) / features_df['Close'] - 1
        
        # Time features
        features_df['day_of_week'] = features_df['time'].dt.dayofweek
        features_df['month'] = features_df['time'].dt.month
        
        return features_df
    
    def normalize_features(self, data, feature_columns):
        """Chuẩn hóa features đơn giản (z-score)"""
        normalized_data = []
        stats = {}
        
        for col in feature_columns:
            if col in data.columns:
                values = data[col].dropna().tolist()
                if len(values) > 1:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 1
                    std_val = max(std_val, 1e-8)  # Tránh chia cho 0
                    
                    normalized_col = [(x - mean_val) / std_val for x in data[col].fillna(mean_val)]
                    stats[col] = {'mean': mean_val, 'std': std_val}
                else:
                    normalized_col = [0] * len(data)
                    stats[col] = {'mean': 0, 'std': 1}
                
                normalized_data.append(normalized_col)
            else:
                normalized_data.append([0] * len(data))
                stats[col] = {'mean': 0, 'std': 1}
        
        return np.array(normalized_data).T, stats
    
    def simple_linear_regression(self, X, y):
        """Hồi quy tuyến tính đơn giản với multiple features"""
        # Thêm bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: theta = (X^T * X)^(-1) * X^T * y
        try:
            XtX = np.dot(X_with_bias.T, X_with_bias)
            XtX_inv = np.linalg.inv(XtX + np.eye(XtX.shape[0]) * 1e-6)  # Ridge regularization
            Xty = np.dot(X_with_bias.T, y)
            theta = np.dot(XtX_inv, Xty)
            return theta
        except:
            # Fallback: simple average
            return np.zeros(X.shape[1] + 1)
    
    def predict_linear(self, X, theta):
        """Dự đoán với mô hình tuyến tính"""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return np.dot(X_with_bias, theta)
    
    def train_model(self, symbol):
        """Huấn luyện mô hình đơn giản cho một mã"""
        print(f"\n🤖 Đang huấn luyện mô hình cho {symbol}...")
        
        if symbol not in self.data:
            print(f"❌ Không có dữ liệu cho {symbol}")
            return False
        
        # Chuẩn bị dữ liệu
        df = self.create_features(self.data[symbol])
        
        # Chọn features
        feature_columns = [
            'price_change_1d', 'price_change_3d', 'price_change_5d',
            'volume_change', 'hl_ratio', 'oc_ratio',
            'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
            'close_lag1', 'close_lag2', 'volume_lag1',
            'day_of_week', 'month'
        ]
        
        # Thêm RSI nếu có
        if 'RSI' in df.columns:
            feature_columns.append('RSI')
        
        # Loại bỏ dữ liệu null
        df_clean = df.dropna(subset=feature_columns + ['target_price_change'])
        
        if len(df_clean) < 30:
            print(f"❌ Không đủ dữ liệu để huấn luyện {symbol}")
            return False
        
        # Chia train/test theo thời gian
        split_idx = int(len(df_clean) * 0.8)
        train_data = df_clean.iloc[:split_idx]
        test_data = df_clean.iloc[split_idx:]
        
        # Chuẩn hóa features
        X_train, train_stats = self.normalize_features(train_data, feature_columns)
        y_train = train_data['target_price_change'].values
        
        # Huấn luyện mô hình
        theta = self.simple_linear_regression(X_train, y_train)
        
        # Test
        X_test, _ = self.normalize_features(test_data, feature_columns)
        # Sử dụng train_stats để normalize test data
        X_test_normalized = []
        for i, col in enumerate(feature_columns):
            if col in train_stats:
                mean_val = train_stats[col]['mean']
                std_val = train_stats[col]['std']
                test_values = test_data[col].fillna(mean_val).tolist()
                normalized_values = [(x - mean_val) / std_val for x in test_values]
                X_test_normalized.append(normalized_values)
            else:
                X_test_normalized.append([0] * len(test_data))
        
        X_test = np.array(X_test_normalized).T
        y_test = test_data['target_price_change'].values
        
        # Dự đoán
        y_pred = self.predict_linear(X_test, theta)
        
        # Đánh giá
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Lưu mô hình
        self.models[symbol] = {
            'theta': theta,
            'feature_columns': feature_columns,
            'stats': train_stats,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        # Lưu kết quả dự đoán
        self.predictions[symbol] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'dates': test_data['time'].values,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"✅ Hoàn thành huấn luyện {symbol}")
        print(f"   MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        
        return True
    
    def train_all_models(self):
        """Huấn luyện mô hình cho tất cả các mã"""
        print("🚀 Bắt đầu huấn luyện mô hình...")
        
        for symbol in self.symbols:
            self.train_model(symbol)
    
    def predict_next_day(self, symbol):
        """Dự báo ngày tiếp theo"""
        if symbol not in self.models:
            print(f"❌ Chưa có mô hình cho {symbol}")
            return None
        
        model = self.models[symbol]
        df = self.create_features(self.data[symbol])
        
        # Lấy dữ liệu mới nhất
        latest_data = df.iloc[-1:]
        feature_columns = model['feature_columns']
        
        # Chuẩn hóa
        X_latest = []
        for col in feature_columns:
            if col in model['stats']:
                mean_val = model['stats'][col]['mean']
                std_val = model['stats'][col]['std']
                value = latest_data[col].iloc[0] if col in latest_data.columns else mean_val
                normalized_value = (value - mean_val) / std_val
                X_latest.append(normalized_value)
            else:
                X_latest.append(0)
        
        X_latest = np.array([X_latest])
        
        # Dự đoán
        prediction = self.predict_linear(X_latest, model['theta'])[0]
        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + prediction)
        
        return {
            'current_price': current_price,
            'predicted_change': prediction,
            'predicted_price': predicted_price
        }
    
    def plot_results_simple(self):
        """Vẽ biểu đồ kết quả đơn giản với matplotlib"""
        if not self.predictions:
            print("❌ Không có dữ liệu để vẽ")
            return
        
        n_symbols = len(self.predictions)
        fig, axes = plt.subplots(n_symbols, 2, figsize=(15, 5 * n_symbols))
        
        if n_symbols == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green']
        
        for idx, (symbol, pred_data) in enumerate(self.predictions.items()):
            color = colors[idx % len(colors)]
            
            # Biểu đồ dự báo vs thực tế
            ax1 = axes[idx][0]
            ax1.plot(pred_data['y_true'], label='Thực tế', color=color, alpha=0.8)
            ax1.plot(pred_data['y_pred'], label='Dự báo', color=color, linestyle='--', alpha=0.8)
            ax1.set_title(f'{symbol} - Dự báo vs Thực tế')
            ax1.set_xlabel('Thời gian')
            ax1.set_ylabel('Thay đổi giá (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Biểu đồ sai số
            ax2 = axes[idx][1]
            errors = pred_data['y_pred'] - pred_data['y_true']
            ax2.plot(errors, color=color, alpha=0.8)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title(f'{symbol} - Sai số')
            ax2.set_xlabel('Thời gian')
            ax2.set_ylabel('Sai số')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Đã lưu biểu đồ tại prediction_results.png")
    
    def plot_error_analysis(self):
        """Phân tích sai số"""
        if not self.predictions:
            print("❌ Không có dữ liệu để phân tích")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        all_errors = []
        all_true = []
        all_pred = []
        
        colors = ['blue', 'red', 'green']
        
        for idx, (symbol, pred_data) in enumerate(self.predictions.items()):
            color = colors[idx % len(colors)]
            errors = pred_data['y_pred'] - pred_data['y_true']
            
            all_errors.extend(errors)
            all_true.extend(pred_data['y_true'])
            all_pred.extend(pred_data['y_pred'])
            
            # Histogram sai số
            axes[0, 0].hist(errors, bins=30, alpha=0.7, label=symbol, color=color)
            
            # Scatter plot
            axes[0, 1].scatter(pred_data['y_true'], pred_data['y_pred'], 
                             alpha=0.6, label=symbol, color=color)
        
        # Hoàn thiện histogram
        axes[0, 0].set_title('Phân Phối Sai Số')
        axes[0, 0].set_xlabel('Sai số')
        axes[0, 0].set_ylabel('Tần suất')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hoàn thiện scatter plot
        axes[0, 1].set_title('Dự báo vs Thực tế')
        axes[0, 1].set_xlabel('Thực tế')
        axes[0, 1].set_ylabel('Dự báo')
        # Thêm đường y=x
        min_val, max_val = min(all_true), max(all_true)
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Độ chính xác theo thời gian
        for idx, (symbol, pred_data) in enumerate(self.predictions.items()):
            color = colors[idx % len(colors)]
            # Rolling accuracy (direction prediction)
            y_true_direction = np.sign(pred_data['y_true'])
            y_pred_direction = np.sign(pred_data['y_pred'])
            accuracy = (y_true_direction == y_pred_direction).astype(int)
            
            # Rolling average của accuracy
            window = min(10, len(accuracy) // 3)
            if window > 1:
                rolling_acc = []
                for i in range(window, len(accuracy)):
                    rolling_acc.append(np.mean(accuracy[i-window:i]))
                axes[1, 0].plot(range(window, len(accuracy)), rolling_acc, 
                               label=f'{symbol}', color=color)
        
        axes[1, 0].set_title('Độ Chính Xác Chiều Theo Thời Gian')
        axes[1, 0].set_xlabel('Thời gian')
        axes[1, 0].set_ylabel('Độ chính xác')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot R² scores
        r2_scores = [pred_data['r2'] for pred_data in self.predictions.values()]
        symbols_list = list(self.predictions.keys())
        
        axes[1, 1].bar(symbols_list, r2_scores, color=colors[:len(symbols_list)])
        axes[1, 1].set_title('R² Score theo từng mã')
        axes[1, 1].set_xlabel('Mã chứng khoán')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Đã lưu phân tích sai số tại error_analysis.png")
    
    def save_results_json(self, filename="prediction_results.json"):
        """Lưu kết quả dưới dạng JSON"""
        results = {}
        
        for symbol in self.symbols:
            if symbol in self.models and symbol in self.predictions:
                pred_data = self.predictions[symbol]
                model_data = self.models[symbol]
                
                # Dự báo ngày tiếp theo
                next_day_pred = self.predict_next_day(symbol)
                
                results[symbol] = {
                    'model_performance': {
                        'mse': float(model_data['mse']),
                        'mae': float(model_data['mae']),
                        'r2': float(model_data['r2'])
                    },
                    'direction_accuracy': float(np.mean(np.sign(pred_data['y_true']) == np.sign(pred_data['y_pred'])) * 100),
                    'next_day_prediction': {
                        'current_price': float(next_day_pred['current_price']) if next_day_pred else None,
                        'predicted_change': float(next_day_pred['predicted_change']) if next_day_pred else None,
                        'predicted_price': float(next_day_pred['predicted_price']) if next_day_pred else None
                    } if next_day_pred else None
                }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Đã lưu kết quả tại {filename}")
    
    def print_summary(self):
        """In tóm tắt kết quả"""
        print("\n" + "="*60)
        print("📋 TÓM TẮT KẾT QUẢ MÔ HÌNH DỰ BÁO SIÊU NHẸ")
        print("="*60)
        
        if not self.predictions:
            print("❌ Không có kết quả dự báo")
            return
        
        for symbol, pred_data in self.predictions.items():
            print(f"\n🏢 {symbol}:")
            print(f"   MSE: {pred_data['mse']:.6f}")
            print(f"   MAE: {pred_data['mae']:.6f}")
            print(f"   R²: {pred_data['r2']:.4f}")
            
            # Độ chính xác chiều
            y_true_direction = np.sign(pred_data['y_true'])
            y_pred_direction = np.sign(pred_data['y_pred'])
            direction_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
            print(f"   Độ chính xác chiều: {direction_accuracy:.1f}%")
            
            # Dự báo ngày tiếp theo
            next_pred = self.predict_next_day(symbol)
            if next_pred:
                print(f"   Giá hiện tại: {next_pred['current_price']:.2f}")
                print(f"   Dự báo thay đổi: {next_pred['predicted_change']:.4f} ({next_pred['predicted_change']*100:.2f}%)")
                print(f"   Giá dự báo: {next_pred['predicted_price']:.2f}")
                direction = "📈 TĂNG" if next_pred['predicted_change'] > 0 else "📉 GIẢM"
                print(f"   Xu hướng: {direction}")


def main():
    """Hàm chính"""
    print("🚀 KHỞI ĐỘNG MÔ HÌNH DỰ BÁO CHỨNG KHOÁN SIÊU NHẸ")
    print("=" * 60)
    
    # Khởi tạo
    predictor = UltraLightStockPredictor(['VCB', 'VNM', 'FPT'])
    
    # Tải dữ liệu
    predictor.load_data()
    
    # Huấn luyện
    predictor.train_all_models()
    
    # In kết quả
    predictor.print_summary()
    
    # Vẽ biểu đồ
    print("\n📊 Đang tạo biểu đồ...")
    predictor.plot_results_simple()
    predictor.plot_error_analysis()
    
    # Lưu kết quả
    predictor.save_results_json()
    
    print("\n✅ HOÀN THÀNH! Kiểm tra các file PNG và JSON đã được tạo.")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
