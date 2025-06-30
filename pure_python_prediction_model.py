#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mô Hình Dự Báo Chứng Khoán Pure Python
Chỉ sử dụng thư viện chuẩn của Python
Author: Vu Dang Khoa
"""

import csv
import json
import os
import math
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

class PurePythonStockPredictor:
    """Mô hình dự báo chứng khoán chỉ dùng Python thuần"""
    
    def __init__(self, symbols=['VCB', 'VNM', 'FPT']):
        self.symbols = symbols
        self.data = {}
        self.models = {}
        self.predictions = {}
        
    def load_csv_data(self, filepath):
        """Đọc file CSV bằng module csv chuẩn"""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Chuyển đổi kiểu dữ liệu
                    processed_row = {}
                    for key, value in row.items():
                        try:
                            # Thử chuyển thành float trước
                            if value and value != '' and key != 'time':
                                processed_row[key] = float(value)
                            else:
                                processed_row[key] = value
                        except ValueError:
                            processed_row[key] = value
                    data.append(processed_row)
            return data
        except Exception as e:
            print(f"❌ Lỗi đọc file {filepath}: {str(e)}")
            return []
    
    def parse_date(self, date_str):
        """Parse date string thành datetime"""
        try:
            return datetime.strptime(date_str.split()[0], '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                return datetime.now()
    
    def load_data(self, data_path="/Users/vudangkhoa/Working/KhoaStock/data/collected_data"):
        """Tải dữ liệu cho tất cả các mã"""
        print("📊 Đang tải dữ liệu...")
        
        for symbol in self.symbols:
            try:
                # Đọc dữ liệu technical
                tech_file = f"{data_path}/technical/{symbol}_technical.csv"
                if os.path.exists(tech_file):
                    tech_data = self.load_csv_data(tech_file)
                    
                    # Sắp xếp theo thời gian
                    for row in tech_data:
                        row['datetime'] = self.parse_date(row['time'])
                    
                    tech_data.sort(key=lambda x: x['datetime'])
                    self.data[symbol] = tech_data
                    print(f"✅ Đã tải {len(tech_data)} dòng dữ liệu cho {symbol}")
                else:
                    print(f"❌ Không tìm thấy file {tech_file}")
                    
            except Exception as e:
                print(f"❌ Lỗi khi tải dữ liệu {symbol}: {str(e)}")
    
    def calculate_moving_average(self, values, window):
        """Tính trung bình động"""
        result = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(None)
            else:
                avg = sum(values[i-window+1:i+1]) / window
                result.append(avg)
        return result
    
    def calculate_percentage_change(self, values, period=1):
        """Tính phần trăm thay đổi"""
        result = []
        for i in range(len(values)):
            if i < period:
                result.append(0)
            else:
                if values[i-period] != 0:
                    change = (values[i] - values[i-period]) / values[i-period]
                    result.append(change)
                else:
                    result.append(0)
        return result
    
    def calculate_rsi(self, prices, period=14):
        """Tính RSI"""
        if len(prices) < period + 1:
            return [50] * len(prices)
        
        deltas = []
        for i in range(1, len(prices)):
            deltas.append(prices[i] - prices[i-1])
        
        gains = [max(d, 0) for d in deltas]
        losses = [abs(min(d, 0)) for d in deltas]
        
        rsi_values = [50] * (period + 1)
        
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
    
    def create_features(self, data):
        """Tạo features từ dữ liệu"""
        if not data:
            return []
        
        # Lấy các cột số
        close_prices = [row['Close'] for row in data if row['Close'] is not None]
        volumes = [row['Volume'] for row in data if row['Volume'] is not None]
        highs = [row['High'] for row in data if row['High'] is not None]
        lows = [row['Low'] for row in data if row['Low'] is not None]
        opens = [row['Open'] for row in data if row['Open'] is not None]
        
        # Features cơ bản
        price_change_1d = self.calculate_percentage_change(close_prices, 1)
        price_change_3d = self.calculate_percentage_change(close_prices, 3)
        price_change_5d = self.calculate_percentage_change(close_prices, 5)
        volume_change = self.calculate_percentage_change(volumes, 1)
        
        # Moving averages
        ma5 = self.calculate_moving_average(close_prices, 5)
        ma10 = self.calculate_moving_average(close_prices, 10)
        ma20 = self.calculate_moving_average(close_prices, 20)
        
        # RSI
        rsi_values = self.calculate_rsi(close_prices)
        
        # Tạo dataset với features
        features_data = []
        for i in range(len(data)):
            if i < 20:  # Bỏ qua các dòng đầu vì thiếu MA20
                continue
                
            row = data[i]
            
            # Tính các ratio
            hl_ratio = (highs[i] - lows[i]) / close_prices[i] if close_prices[i] != 0 else 0
            oc_ratio = (close_prices[i] - opens[i]) / opens[i] if opens[i] != 0 else 0
            
            price_ma5_ratio = close_prices[i] / ma5[i] if ma5[i] is not None and ma5[i] != 0 else 1
            price_ma10_ratio = close_prices[i] / ma10[i] if ma10[i] is not None and ma10[i] != 0 else 1
            price_ma20_ratio = close_prices[i] / ma20[i] if ma20[i] is not None and ma20[i] != 0 else 1
            
            # Lag features
            close_lag1 = close_prices[i-1] if i > 0 else close_prices[i]
            close_lag2 = close_prices[i-2] if i > 1 else close_prices[i]
            volume_lag1 = volumes[i-1] if i > 0 else volumes[i]
            
            # Time features
            day_of_week = row['datetime'].weekday()
            month = row['datetime'].month
            
            # Target (next day price change)
            target = 0
            if i < len(close_prices) - 1:
                target = (close_prices[i+1] - close_prices[i]) / close_prices[i]
            
            features = {
                'price_change_1d': price_change_1d[i],
                'price_change_3d': price_change_3d[i],
                'price_change_5d': price_change_5d[i],
                'volume_change': volume_change[i],
                'hl_ratio': hl_ratio,
                'oc_ratio': oc_ratio,
                'price_ma5_ratio': price_ma5_ratio,
                'price_ma10_ratio': price_ma10_ratio,
                'price_ma20_ratio': price_ma20_ratio,
                'close_lag1': close_lag1,
                'close_lag2': close_lag2,
                'volume_lag1': volume_lag1,
                'rsi': rsi_values[i] if i < len(rsi_values) else 50,
                'day_of_week': day_of_week,
                'month': month,
                'target': target,
                'current_price': close_prices[i],
                'datetime': row['datetime']
            }
            
            features_data.append(features)
        
        return features_data
    
    def normalize_data(self, data, feature_names):
        """Chuẩn hóa dữ liệu (z-score)"""
        normalized_data = []
        stats = {}
        
        for feature in feature_names:
            values = [row[feature] for row in data if feature in row]
            if len(values) > 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                std_val = max(std_val, 1e-8)  # Tránh chia cho 0
                stats[feature] = {'mean': mean_val, 'std': std_val}
            else:
                stats[feature] = {'mean': 0, 'std': 1}
        
        for row in data:
            normalized_row = row.copy()
            for feature in feature_names:
                if feature in row and feature in stats:
                    normalized_row[feature] = (row[feature] - stats[feature]['mean']) / stats[feature]['std']
            normalized_data.append(normalized_row)
        
        return normalized_data, stats
    
    def matrix_multiply(self, A, B):
        """Nhân ma trận"""
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                sum_val = 0
                for k in range(len(B)):
                    sum_val += A[i][k] * B[k][j]
                row.append(sum_val)
            result.append(row)
        return result
    
    def matrix_transpose(self, matrix):
        """Chuyển vị ma trận"""
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    def matrix_inverse_2x2(self, matrix):
        """Nghịch đảo ma trận 2x2"""
        a, b = matrix[0]
        c, d = matrix[1]
        det = a*d - b*c
        if abs(det) < 1e-10:
            return [[1, 0], [0, 1]]  # Identity matrix nếu không nghịch đảo được
        return [[d/det, -b/det], [-c/det, a/det]]
    
    def simple_linear_regression(self, X, y):
        """Hồi quy tuyến tính đơn giản"""
        n = len(X)
        if n == 0:
            return [0, 0]
        
        # Thêm bias term (cột 1)
        X_with_bias = [[1] + row for row in X]
        
        # Simple implementation cho trường hợp đặc biệt
        if len(X[0]) == 1:  # Chỉ có 1 feature
            # y = a*x + b
            sum_x = sum(row[0] for row in X)
            sum_y = sum(y)
            sum_xy = sum(X[i][0] * y[i] for i in range(n))
            sum_x2 = sum(row[0]**2 for row in X)
            
            if n * sum_x2 - sum_x**2 != 0:
                a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
                b = (sum_y - a * sum_x) / n
                return [b, a]  # [bias, weight]
            else:
                return [sum_y/n, 0]  # Chỉ có bias
        
        # Cho nhiều features, dùng phương pháp đơn giản
        # Tính trung bình của target
        mean_y = sum(y) / len(y)
        
        # Tính correlation với từng feature
        weights = [mean_y]  # bias
        for j in range(len(X[0])):
            feature_values = [row[j] for row in X]
            if len(set(feature_values)) > 1:  # Feature có variance
                # Tính correlation đơn giản
                mean_x = sum(feature_values) / len(feature_values)
                numerator = sum((feature_values[i] - mean_x) * (y[i] - mean_y) for i in range(len(y)))
                denominator = sum((x - mean_x)**2 for x in feature_values)
                if denominator != 0:
                    weight = numerator / denominator * 0.1  # Scale down để tránh overfitting
                else:
                    weight = 0
            else:
                weight = 0
            weights.append(weight)
        
        return weights
    
    def predict(self, X, weights):
        """Dự đoán với mô hình tuyến tính"""
        predictions = []
        for row in X:
            pred = weights[0]  # bias
            for i, feature in enumerate(row):
                if i + 1 < len(weights):
                    pred += feature * weights[i + 1]
            predictions.append(pred)
        return predictions
    
    def train_model(self, symbol):
        """Huấn luyện mô hình cho một mã"""
        print(f"\n🤖 Đang huấn luyện mô hình cho {symbol}...")
        
        if symbol not in self.data:
            print(f"❌ Không có dữ liệu cho {symbol}")
            return False
        
        # Tạo features
        features_data = self.create_features(self.data[symbol])
        
        if len(features_data) < 30:
            print(f"❌ Không đủ dữ liệu để huấn luyện {symbol}")
            return False
        
        # Loại bỏ dòng cuối (không có target)
        features_data = features_data[:-1]
        
        # Chia train/test
        split_idx = int(len(features_data) * 0.8)
        train_data = features_data[:split_idx]
        test_data = features_data[split_idx:]
        
        # Feature names
        feature_names = [
            'price_change_1d', 'price_change_3d', 'price_change_5d',
            'volume_change', 'hl_ratio', 'oc_ratio',
            'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
            'rsi', 'day_of_week', 'month'
        ]
        
        # Chuẩn hóa dữ liệu train
        train_normalized, stats = self.normalize_data(train_data, feature_names)
        
        # Chuẩn bị X, y cho training
        X_train = []
        y_train = []
        for row in train_normalized:
            features = [row[f] for f in feature_names]
            X_train.append(features)
            y_train.append(row['target'])
        
        # Huấn luyện
        weights = self.simple_linear_regression(X_train, y_train)
        
        # Chuẩn hóa dữ liệu test với stats từ train
        test_normalized = []
        for row in test_data:
            normalized_row = row.copy()
            for feature in feature_names:
                if feature in stats:
                    normalized_row[feature] = (row[feature] - stats[feature]['mean']) / stats[feature]['std']
            test_normalized.append(normalized_row)
        
        # Test
        X_test = []
        y_test = []
        for row in test_normalized:
            features = [row[f] for f in feature_names]
            X_test.append(features)
            y_test.append(row['target'])
        
        y_pred = self.predict(X_test, weights)
        
        # Đánh giá
        mse = sum((y_test[i] - y_pred[i])**2 for i in range(len(y_test))) / len(y_test)
        mae = sum(abs(y_test[i] - y_pred[i]) for i in range(len(y_test))) / len(y_test)
        
        # R-squared
        mean_y = sum(y_test) / len(y_test)
        ss_res = sum((y_test[i] - y_pred[i])**2 for i in range(len(y_test)))
        ss_tot = sum((y - mean_y)**2 for y in y_test)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Lưu mô hình
        self.models[symbol] = {
            'weights': weights,
            'feature_names': feature_names,
            'stats': stats,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        # Lưu predictions
        self.predictions[symbol] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'test_data': test_data,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"✅ Hoàn thành huấn luyện {symbol}")
        print(f"   MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        
        return True
    
    def train_all_models(self):
        """Huấn luyện tất cả mô hình"""
        print("🚀 Bắt đầu huấn luyện mô hình...")
        
        for symbol in self.symbols:
            self.train_model(symbol)
    
    def predict_next_day(self, symbol):
        """Dự báo ngày tiếp theo"""
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]
        features_data = self.create_features(self.data[symbol])
        
        if not features_data:
            return None
        
        # Lấy dữ liệu mới nhất
        latest_data = features_data[-1]
        
        # Chuẩn hóa
        normalized_features = []
        for feature in model['feature_names']:
            if feature in model['stats']:
                value = latest_data[feature]
                mean_val = model['stats'][feature]['mean']
                std_val = model['stats'][feature]['std']
                normalized_value = (value - mean_val) / std_val
                normalized_features.append(normalized_value)
            else:
                normalized_features.append(0)
        
        # Dự đoán
        prediction = self.predict([normalized_features], model['weights'])[0]
        current_price = latest_data['current_price']
        predicted_price = current_price * (1 + prediction)
        
        return {
            'current_price': current_price,
            'predicted_change': prediction,
            'predicted_price': predicted_price
        }
    
    def save_results(self, filename="prediction_results.json"):
        """Lưu kết quả"""
        results = {}
        
        for symbol in self.symbols:
            if symbol in self.models and symbol in self.predictions:
                pred_data = self.predictions[symbol]
                model_data = self.models[symbol]
                
                # Direction accuracy
                y_true_direction = [1 if y > 0 else -1 for y in pred_data['y_true']]
                y_pred_direction = [1 if y > 0 else -1 for y in pred_data['y_pred']]
                direction_accuracy = sum(1 for i in range(len(y_true_direction)) 
                                       if y_true_direction[i] == y_pred_direction[i]) / len(y_true_direction) * 100
                
                # Next day prediction
                next_pred = self.predict_next_day(symbol)
                
                results[symbol] = {
                    'model_performance': {
                        'mse': model_data['mse'],
                        'mae': model_data['mae'],
                        'r2': model_data['r2'],
                        'direction_accuracy': direction_accuracy
                    },
                    'next_day_prediction': next_pred
                }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 Đã lưu kết quả tại {filename}")
    
    def create_simple_plot_data(self):
        """Tạo dữ liệu để vẽ biểu đồ (dạng CSV)"""
        for symbol, pred_data in self.predictions.items():
            # Tạo file CSV cho từng mã
            filename = f"{symbol}_prediction_plot.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Index', 'True_Value', 'Predicted_Value', 'Error'])
                
                for i in range(len(pred_data['y_true'])):
                    true_val = pred_data['y_true'][i]
                    pred_val = pred_data['y_pred'][i]
                    error = pred_val - true_val
                    writer.writerow([i, true_val, pred_val, error])
            
            print(f"📊 Đã tạo dữ liệu biểu đồ cho {symbol} tại {filename}")
    
    def print_summary(self):
        """In tóm tắt kết quả"""
        print("\n" + "="*60)
        print("📋 TÓM TẮT KẾT QUẢ MÔ HÌNH DỰ BÁO (PURE PYTHON)")
        print("="*60)
        
        if not self.predictions:
            print("❌ Không có kết quả dự báo")
            return
        
        for symbol, pred_data in self.predictions.items():
            print(f"\n🏢 {symbol}:")
            print(f"   MSE: {pred_data['mse']:.6f}")
            print(f"   MAE: {pred_data['mae']:.6f}")
            print(f"   R²: {pred_data['r2']:.4f}")
            
            # Direction accuracy
            y_true_direction = [1 if y > 0 else -1 for y in pred_data['y_true']]
            y_pred_direction = [1 if y > 0 else -1 for y in pred_data['y_pred']]
            direction_accuracy = sum(1 for i in range(len(y_true_direction)) 
                                   if y_true_direction[i] == y_pred_direction[i]) / len(y_true_direction) * 100
            print(f"   Độ chính xác chiều: {direction_accuracy:.1f}%")
            
            # Next day prediction
            next_pred = self.predict_next_day(symbol)
            if next_pred:
                print(f"   Giá hiện tại: {next_pred['current_price']:.2f}")
                print(f"   Dự báo thay đổi: {next_pred['predicted_change']:.4f} ({next_pred['predicted_change']*100:.2f}%)")
                print(f"   Giá dự báo: {next_pred['predicted_price']:.2f}")
                direction = "📈 TĂNG" if next_pred['predicted_change'] > 0 else "📉 GIẢM"
                print(f"   Xu hướng: {direction}")


def main():
    """Hàm chính"""
    print("🚀 KHỞI ĐỘNG MÔ HÌNH DỰ BÁO CHỨNG KHOÁN (PURE PYTHON)")
    print("=" * 60)
    
    # Khởi tạo
    predictor = PurePythonStockPredictor(['VCB', 'VNM', 'FPT'])
    
    # Tải dữ liệu
    predictor.load_data()
    
    # Huấn luyện
    predictor.train_all_models()
    
    # In kết quả
    predictor.print_summary()
    
    # Lưu kết quả
    predictor.save_results()
    
    # Tạo dữ liệu biểu đồ
    predictor.create_simple_plot_data()
    
    print("\n✅ HOÀN THÀNH! Kiểm tra các file JSON và CSV đã được tạo.")
    print("📊 Dùng Excel hoặc Google Sheets để vẽ biểu đồ từ file CSV.")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
