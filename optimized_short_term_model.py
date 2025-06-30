#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import math
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

class OptimizedShortTermPredictor:
    
    def __init__(self, symbols=['VCB', 'VNM', 'FPT']):
        self.symbols = symbols
        self.data = {}
        self.models = {}
        self.predictions = {}
        self.results = {}
        
    def load_csv_data(self, filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    processed_row = {}
                    for key, value in row.items():
                        try:
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
        """Parse date string"""
        try:
            return datetime.strptime(date_str.split()[0], '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                return datetime.now()
    
    def load_data(self, data_path="/Users/vudangkhoa/Working/KhoaStock/data/collected_data"):
        print("📊 Đang tải dữ liệu...")
        
        for symbol in self.symbols:
            try:
                tech_file = f"{data_path}/technical/{symbol}_technical.csv"
                if os.path.exists(tech_file):
                    tech_data = self.load_csv_data(tech_file)
                    
                    for row in tech_data:
                        row['datetime'] = self.parse_date(row['time'])
                    
                    tech_data.sort(key=lambda x: x['datetime'])
                    self.data[symbol] = tech_data
                    print(f"✅ Đã tải {len(tech_data)} dòng dữ liệu cho {symbol}")
                else:
                    print(f"❌ Không tìm thấy file {tech_file}")
                    
            except Exception as e:
                print(f"❌ Lỗi khi tải dữ liệu {symbol}: {str(e)}")
    
    def calculate_momentum_features(self, prices, volumes):
        features = {}
        
        if len(prices) < 5:
            return features
        
        # Price momentum (tốc độ thay đổi giá)
        features['price_momentum_1'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        features['price_momentum_2'] = (prices[-1] - prices[-3]) / prices[-3] if len(prices) > 2 and prices[-3] != 0 else 0
        features['price_momentum_3'] = (prices[-1] - prices[-4]) / prices[-4] if len(prices) > 3 and prices[-4] != 0 else 0
        
        # Volume momentum
        if len(volumes) >= 3:
            features['volume_momentum_1'] = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
            features['volume_momentum_2'] = (volumes[-1] - volumes[-3]) / volumes[-3] if volumes[-3] != 0 else 0
        
        # Price volatility (độ biến động)
        recent_prices = prices[-5:]
        if len(recent_prices) > 1:
            mean_price = sum(recent_prices) / len(recent_prices)
            variance = sum((p - mean_price)**2 for p in recent_prices) / len(recent_prices)
            features['price_volatility'] = math.sqrt(variance) / mean_price if mean_price != 0 else 0
        
        # Volume surge (đột biến khối lượng)
        if len(volumes) >= 5:
            recent_volumes = volumes[-5:]
            mean_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1])
            features['volume_surge'] = volumes[-1] / mean_volume if mean_volume != 0 else 1
        
        return features
    
    def calculate_micro_technical_indicators(self, data):
        closes = [row['Close'] for row in data]
        highs = [row['High'] for row in data]
        lows = [row['Low'] for row in data]
        volumes = [row['Volume'] for row in data]
        
        indicators = []
        
        for i in range(len(data)):
            if i < 5:  # Cần ít nhất 5 điểm dữ liệu
                indicators.append({})
                continue
            
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            current_volume = volumes[i]
            
            # Micro patterns (các mẫu hình vi mô)
            micro_indicators = {}
            
            # 1. Candle body ratio
            open_price = data[i]['Open']
            body_size = abs(current_close - open_price)
            total_range = current_high - current_low
            micro_indicators['body_ratio'] = body_size / total_range if total_range != 0 else 0
            
            # 2. Upper/Lower shadow ratio
            if current_close > open_price:  # Green candle
                upper_shadow = current_high - current_close
                lower_shadow = open_price - current_low
            else:  # Red candle
                upper_shadow = current_high - open_price
                lower_shadow = current_close - current_low
            
            micro_indicators['upper_shadow_ratio'] = upper_shadow / total_range if total_range != 0 else 0
            micro_indicators['lower_shadow_ratio'] = lower_shadow / total_range if total_range != 0 else 0
            
            # 3. Price position in recent range
            recent_highs = highs[max(0, i-4):i+1]
            recent_lows = lows[max(0, i-4):i+1]
            if recent_highs and recent_lows:
                range_high = max(recent_highs)
                range_low = min(recent_lows)
                if range_high != range_low:
                    micro_indicators['price_position'] = (current_close - range_low) / (range_high - range_low)
                else:
                    micro_indicators['price_position'] = 0.5
            
            # 4. Volume pressure
            recent_volumes = volumes[max(0, i-4):i+1]
            if recent_volumes:
                avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1]) if len(recent_volumes) > 1 else recent_volumes[0]
                micro_indicators['volume_pressure'] = current_volume / avg_volume if avg_volume != 0 else 1
            
            # 5. Price acceleration
            if i >= 2:
                price_change_1 = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
                price_change_2 = (closes[i-1] - closes[i-2]) / closes[i-2] if closes[i-2] != 0 else 0
                micro_indicators['price_acceleration'] = price_change_1 - price_change_2
            
            # 6. Momentum features
            momentum_features = self.calculate_momentum_features(closes[:i+1], volumes[:i+1])
            micro_indicators.update(momentum_features)
            
            indicators.append(micro_indicators)
        
        return indicators
    
    def create_short_term_features(self, data):
        """Tạo features chuyên dụng cho dự báo ngắn hạn"""
        if not data:
            return []
        
        # Tính micro indicators
        micro_indicators = self.calculate_micro_technical_indicators(data)
        
        closes = [row['Close'] for row in data]
        features_data = []
        
        for i in range(len(data)):
            if i < 10:  # Bỏ qua 10 điểm đầu
                continue
            
            # Target cho ngắn hạn (thay đổi trong 1-3 phiên tới)
            targets = {}
            for horizon in [1, 2, 3]:
                if i + horizon < len(closes):
                    future_price = closes[i + horizon]
                    current_price = closes[i]
                    targets[f'target_{horizon}'] = (future_price - current_price) / current_price
                else:
                    targets[f'target_{horizon}'] = 0
            
            # Features từ micro indicators
            features = micro_indicators[i].copy()
            
            # Thêm lag features
            for lag in [1, 2, 3]:
                if i >= lag:
                    features[f'close_lag_{lag}'] = closes[i-lag]
                    if closes[i-lag] != 0:
                        features[f'return_lag_{lag}'] = (closes[i] - closes[i-lag]) / closes[i-lag]
            
            # Time features
            features['day_of_week'] = data[i]['datetime'].weekday()
            features['hour'] = data[i]['datetime'].hour if hasattr(data[i]['datetime'], 'hour') else 12
            
            # Combine features and targets
            features.update(targets)
            features['current_price'] = closes[i]
            features['datetime'] = data[i]['datetime']
            
            features_data.append(features)
        
        return features_data
    
    def ensemble_predict(self, X, models):
        """Ensemble prediction từ nhiều mô hình"""
        predictions = []
        
        for row in X:
            pred_sum = 0
            model_count = 0
            
            for model in models:
                if 'weights' in model:
                    weights = model['weights']
                    pred = weights[0]  # bias
                    for j, feature_val in enumerate(row):
                        if j + 1 < len(weights):
                            pred += feature_val * weights[j + 1]
                    pred_sum += pred
                    model_count += 1
            
            final_pred = pred_sum / model_count if model_count > 0 else 0
            predictions.append(final_pred)
        
        return predictions
    
    def train_ensemble_model(self, symbol):
        """Huấn luyện ensemble model cho dự báo ngắn hạn"""
        print(f"\n🤖 Đang huấn luyện ensemble model cho {symbol}...")
        
        if symbol not in self.data:
            print(f"❌ Không có dữ liệu cho {symbol}")
            return False
        
        # Tạo features cho ngắn hạn
        features_data = self.create_short_term_features(self.data[symbol])
        
        if len(features_data) < 30:
            print(f"❌ Không đủ dữ liệu cho {symbol}")
            return False
        
        # Loại bỏ 3 dòng cuối (không có target)
        features_data = features_data[:-3]
        
        # Feature names (loại bỏ targets và metadata)
        exclude_keys = {'target_1', 'target_2', 'target_3', 'current_price', 'datetime'}
        feature_names = [k for k in features_data[0].keys() if k not in exclude_keys and features_data[0][k] is not None]
        
        # Chia train/test với tỷ lệ phù hợp cho dữ liệu nhỏ
        split_idx = int(len(features_data) * 0.85)  # Tăng tỷ lệ train
        train_data = features_data[:split_idx]
        test_data = features_data[split_idx:]
        
        models = {}
        
        # Huấn luyện cho từng horizon
        for target_name in ['target_1', 'target_2', 'target_3']:
            # Chuẩn bị dữ liệu
            X_train = []
            y_train = []
            
            for row in train_data:
                features = []
                for fname in feature_names:
                    val = row.get(fname, 0)
                    if val is None:
                        val = 0
                    features.append(val)
                X_train.append(features)
                y_train.append(row[target_name])
            
            # Chuẩn hóa
            feature_stats = {}
            for j, fname in enumerate(feature_names):
                values = [X_train[i][j] for i in range(len(X_train))]
                if len(values) > 1:
                    mean_val = statistics.mean(values)
                    try:
                        std_val = statistics.stdev(values)
                    except:
                        std_val = 1
                    std_val = max(std_val, 1e-8)
                    feature_stats[fname] = {'mean': mean_val, 'std': std_val}
                else:
                    feature_stats[fname] = {'mean': 0, 'std': 1}
            
            # Apply normalization
            X_train_norm = []
            for row in X_train:
                norm_row = []
                for j, val in enumerate(row):
                    fname = feature_names[j]
                    if fname in feature_stats:
                        norm_val = (val - feature_stats[fname]['mean']) / feature_stats[fname]['std']
                        norm_row.append(norm_val)
                    else:
                        norm_row.append(val)
                X_train_norm.append(norm_row)
            
            # Train model
            weights = self.advanced_regression(X_train_norm, y_train)
            
            models[target_name] = {
                'weights': weights,
                'feature_names': feature_names,
                'feature_stats': feature_stats
            }
        
        # Test ensemble
        X_test = []
        y_test = {target: [] for target in ['target_1', 'target_2', 'target_3']}
        
        for row in test_data:
            features = []
            for fname in feature_names:
                val = row.get(fname, 0)
                if val is None:
                    val = 0
                features.append(val)
            X_test.append(features)
            
            for target in ['target_1', 'target_2', 'target_3']:
                y_test[target].append(row[target])
        
        # Normalize test data
        X_test_norm = []
        for row in X_test:
            norm_row = []
            for j, val in enumerate(row):
                fname = feature_names[j]
                stats = models['target_1']['feature_stats'][fname]
                norm_val = (val - stats['mean']) / stats['std']
                norm_row.append(norm_val)
            X_test_norm.append(norm_row)
        
        # Predict and evaluate
        results = {}
        for target_name in ['target_1', 'target_2', 'target_3']:
            model = models[target_name]
            y_pred = self.predict(X_test_norm, model['weights'])
            
            # Metrics
            y_true = y_test[target_name]
            mse = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true))) / len(y_true)
            mae = sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)
            
            # Direction accuracy
            direction_acc = sum(1 for i in range(len(y_true)) 
                              if (y_true[i] > 0) == (y_pred[i] > 0)) / len(y_true) * 100
            
            results[target_name] = {
                'mse': mse,
                'mae': mae,
                'direction_accuracy': direction_acc,
                'y_true': y_true,
                'y_pred': y_pred
            }
        
        # Lưu models và results
        self.models[symbol] = models
        self.predictions[symbol] = results
        
        # In kết quả
        print(f"✅ Hoàn thành ensemble model {symbol}")
        for target_name, result in results.items():
            horizon = target_name.split('_')[1]
            print(f"   Horizon {horizon}: MSE={result['mse']:.6f}, Direction={result['direction_accuracy']:.1f}%")
        
        return True
    
    def advanced_regression(self, X, y):
        """Hồi quy với regularization đơn giản"""
        if not X or not y:
            return [0]
        
        n_features = len(X[0])
        n_samples = len(X)
        
        # Ridge regression đơn giản
        alpha = 0.01  # Regularization parameter
        
        # Thêm bias term
        X_with_bias = [[1] + row for row in X]
        
        # Normal equation với regularization: (X^T * X + alpha * I)^-1 * X^T * y
        # Simplified implementation
        
        weights = [0] * (n_features + 1)  # +1 for bias
        
        # Gradient descent đơn giản
        learning_rate = 0.01
        n_iterations = 1000
        
        for iteration in range(n_iterations):
            # Forward pass
            predictions = []
            for i in range(n_samples):
                pred = weights[0]  # bias
                for j in range(n_features):
                    pred += weights[j + 1] * X[i][j]
                predictions.append(pred)
            
            # Calculate gradients
            gradients = [0] * (n_features + 1)
            
            for i in range(n_samples):
                error = predictions[i] - y[i]
                gradients[0] += error  # bias gradient
                for j in range(n_features):
                    gradients[j + 1] += error * X[i][j]
            
            # Update weights with regularization
            for j in range(len(weights)):
                gradient = gradients[j] / n_samples
                if j > 0:  # Add regularization (not for bias)
                    gradient += alpha * weights[j]
                weights[j] -= learning_rate * gradient
            
            # Early stopping
            if iteration % 100 == 0:
                learning_rate *= 0.99  # Decay learning rate
        
        return weights
    
    def predict(self, X, weights):
        """Dự đoán với weights"""
        predictions = []
        for row in X:
            pred = weights[0]  # bias
            for i, feature in enumerate(row):
                if i + 1 < len(weights):
                    pred += feature * weights[i + 1]
            predictions.append(pred)
        return predictions
    
    def train_all_models(self):
        """Huấn luyện tất cả mô hình"""
        print("🚀 Bắt đầu huấn luyện ensemble models...")
        for symbol in self.symbols:
            self.train_ensemble_model(symbol)
    
    def predict_short_term(self, symbol):
        """Dự báo ngắn hạn cho symbol"""
        if symbol not in self.models:
            return None
        
        features_data = self.create_short_term_features(self.data[symbol])
        if not features_data:
            return None
        
        latest_data = features_data[-1]
        models = self.models[symbol]
        
        predictions = {}
        
        for target_name, model in models.items():
            feature_names = model['feature_names']
            feature_stats = model['feature_stats']
            
            # Normalize features
            features = []
            for fname in feature_names:
                val = latest_data.get(fname, 0)
                if val is None:
                    val = 0
                stats = feature_stats[fname]
                norm_val = (val - stats['mean']) / stats['std']
                features.append(norm_val)
            
            # Predict
            pred = self.predict([features], model['weights'])[0]
            predictions[target_name] = pred
        
        current_price = latest_data['current_price']
        
        return {
            'current_price': current_price,
            'horizon_1': {
                'change': predictions['target_1'],
                'price': current_price * (1 + predictions['target_1'])
            },
            'horizon_2': {
                'change': predictions['target_2'],
                'price': current_price * (1 + predictions['target_2'])
            },
            'horizon_3': {
                'change': predictions['target_3'],
                'price': current_price * (1 + predictions['target_3'])
            }
        }
    
    def create_latex_plots(self):
        """Tạo dữ liệu cho biểu đồ LaTeX"""
        print("\n📊 Tạo dữ liệu biểu đồ cho LaTeX...")
        
        # Tạo thư mục figures nếu chưa có
        os.makedirs('model_figuress', exist_ok=True)
        
        for symbol in self.symbols:
            if symbol not in self.predictions:
                continue
            
            print(f"🎨 Tạo biểu đồ cho {symbol}...")
            
            # 1. Prediction vs Actual (cho horizon 1)
            result = self.predictions[symbol]['target_1']
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            # File cho scatter plot
            with open(f'model_figuress/{symbol}_scatter.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['True', 'Predicted'])
                for i in range(len(y_true)):
                    writer.writerow([y_true[i], y_pred[i]])
            
            # File cho time series comparison
            with open(f'model_figuress/{symbol}_timeseries.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Index', 'True', 'Predicted', 'Error'])
                for i in range(len(y_true)):
                    error = y_pred[i] - y_true[i]
                    writer.writerow([i, y_true[i], y_pred[i], error])
            
            # 2. Error distribution
            errors = [y_pred[i] - y_true[i] for i in range(len(y_true))]
            
            # Histogram data
            n_bins = 20
            min_error = min(errors)
            max_error = max(errors)
            bin_width = (max_error - min_error) / n_bins
            
            histogram_data = []
            for i in range(n_bins):
                bin_start = min_error + i * bin_width
                bin_end = bin_start + bin_width
                count = sum(1 for e in errors if bin_start <= e < bin_end)
                bin_center = (bin_start + bin_end) / 2
                histogram_data.append([bin_center, count])
            
            with open(f'model_figuress/{symbol}_histogram.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Error', 'Frequency'])
                for bin_center, count in histogram_data:
                    writer.writerow([bin_center, count])
            
            # 3. Performance metrics over time
            window_size = 10
            rolling_accuracy = []
            rolling_mse = []
            
            for i in range(window_size, len(y_true)):
                window_true = y_true[i-window_size:i]
                window_pred = y_pred[i-window_size:i]
                
                # Direction accuracy
                direction_acc = sum(1 for j in range(len(window_true)) 
                                  if (window_true[j] > 0) == (window_pred[j] > 0)) / len(window_true)
                rolling_accuracy.append(direction_acc)
                
                # MSE
                mse = sum((window_true[j] - window_pred[j])**2 for j in range(len(window_true))) / len(window_true)
                rolling_mse.append(mse)
            
            with open(f'model_figuress/{symbol}_performance.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Index', 'Direction_Accuracy', 'MSE'])
                for i in range(len(rolling_accuracy)):
                    writer.writerow([i + window_size, rolling_accuracy[i], rolling_mse[i]])
            
            # 4. Multi-horizon comparison
            with open(f'model_figuress/{symbol}_multihorizon.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Horizon', 'MSE', 'MAE', 'Direction_Accuracy'])
                for target_name in ['target_1', 'target_2', 'target_3']:
                    if target_name in self.predictions[symbol]:
                        result = self.predictions[symbol][target_name]
                        horizon = target_name.split('_')[1]
                        writer.writerow([horizon, result['mse'], result['mae'], result['direction_accuracy']])
    
    def create_latex_tikz_code(self):
        """Tạo mã LaTeX/TikZ cho biểu đồ"""
        print("\n📝 Tạo mã LaTeX/TikZ...")
        
        latex_code = """\\documentclass{article}
\\usepackage{pgfplots}
\\usepackage{pgfplotstable}
\\pgfplotsset{compat=1.17}

\\begin{document}

"""
        
        for symbol in self.symbols:
            if symbol not in self.predictions:
                continue
            
            latex_code += f"""
% Biểu đồ cho {symbol}
\\section{{{symbol} - Kết Quả Dự Báo}}

% Scatter plot
\\begin{{figure}}[h]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{Giá trị thực tế}},
    ylabel={{Giá trị dự báo}},
    title={{{symbol} - So sánh Dự báo và Thực tế}},
    grid=major,
    width=10cm,
    height=8cm
]
\\addplot[only marks, mark=*, blue] table[x=True, y=Predicted, col sep=comma] {{model_figuress/{symbol}_scatter.csv}};
\\addplot[red, domain=-0.05:0.05] {{x}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{symbol} - Biểu đồ phân tán giữa giá trị dự báo và thực tế}}
\\end{{figure}}

% Time series comparison
\\begin{{figure}}[h]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{Thời gian}},
    ylabel={{Thay đổi giá (\\%)}},
    title={{{symbol} - So sánh chuỗi thời gian}},
    grid=major,
    width=12cm,
    height=8cm,
    legend pos=north west
]
\\addplot[blue, mark=none] table[x=Index, y=True, col sep=comma] {{model_figuress/{symbol}_timeseries.csv}};
\\addplot[red, mark=none] table[x=Index, y=Predicted, col sep=comma] {{model_figuress/{symbol}_timeseries.csv}};
\\legend{{Thực tế, Dự báo}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{symbol} - Chuỗi thời gian thực tế vs dự báo}}
\\end{{figure}}

% Error histogram
\\begin{{figure}}[h]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{Sai số}},
    ylabel={{Tần suất}},
    title={{{symbol} - Phân phối sai số}},
    ybar,
    width=10cm,
    height=8cm
]
\\addplot[fill=blue!50] table[x=Error, y=Frequency, col sep=comma] {{model_figuress/{symbol}_histogram.csv}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{symbol} - Histogram phân phối sai số dự báo}}
\\end{{figure}}

% Performance over time
\\begin{{figure}}[h]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{Thời gian}},
    ylabel={{Độ chính xác}},
    title={{{symbol} - Độ chính xác theo thời gian}},
    grid=major,
    width=12cm,
    height=8cm
]
\\addplot[blue, mark=none] table[x=Index, y=Direction_Accuracy, col sep=comma] {{model_figuress/{symbol}_performance.csv}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{symbol} - Độ chính xác chiều hướng theo thời gian}}
\\end{{figure}}

\\clearpage

"""
        
        latex_code += """
\\end{document}
"""
        
        with open('model_figuress/prediction_plots.tex', 'w') as f:
            f.write(latex_code)
        
        print("✅ Đã tạo file prediction_plots.tex")
    
    def save_comprehensive_results(self):
        """Lưu kết quả chi tiết"""
        results = {
            'model_type': 'Optimized Short-term Ensemble Predictor',
            'target_horizons': ['1 day', '2 days', '3 days'],
            'symbols': {}
        }
        
        for symbol in self.symbols:
            if symbol not in self.predictions:
                continue
            
            symbol_results = {
                'horizons': {}
            }
            
            for target_name in ['target_1', 'target_2', 'target_3']:
                if target_name in self.predictions[symbol]:
                    result = self.predictions[symbol][target_name]
                    horizon = target_name.split('_')[1]
                    
                    symbol_results['horizons'][f'horizon_{horizon}'] = {
                        'mse': result['mse'],
                        'mae': result['mae'],
                        'direction_accuracy': result['direction_accuracy'],
                        'rmse': math.sqrt(result['mse'])
                    }
            
            # Next predictions
            next_pred = self.predict_short_term(symbol)
            if next_pred:
                symbol_results['next_predictions'] = next_pred
            
            results['symbols'][symbol] = symbol_results
        
        with open('model_figuress/comprehensive_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print("💾 Đã lưu kết quả chi tiết tại model_figuress/comprehensive_results.json")
    
    def print_summary(self):
        """In tóm tắt kết quả"""
        print("\n" + "="*80)
        print("📋 TÓM TẮT KẾT QUẢ MÔ HÌNH DỰ BÁO NGẮN HẠN TỐI ƯU")
        print("="*80)
        
        if not self.predictions:
            print("❌ Không có kết quả dự báo")
            return
        
        for symbol in self.symbols:
            if symbol not in self.predictions:
                continue
            
            print(f"\n🏢 {symbol} - Ensemble Model:")
            print("-" * 50)
            
            for target_name in ['target_1', 'target_2', 'target_3']:
                if target_name in self.predictions[symbol]:
                    result = self.predictions[symbol][target_name]
                    horizon = target_name.split('_')[1]
                    
                    print(f"  📊 Horizon {horizon} ngày:")
                    print(f"     MSE: {result['mse']:.6f}")
                    print(f"     MAE: {result['mae']:.6f}")
                    print(f"     RMSE: {math.sqrt(result['mse']):.6f}")
                    print(f"     Độ chính xác chiều: {result['direction_accuracy']:.1f}%")
            
            # Dự báo tiếp theo
            next_pred = self.predict_short_term(symbol)
            if next_pred:
                print(f"\n  🔮 Dự báo tiếp theo:")
                print(f"     Giá hiện tại: {next_pred['current_price']:.2f}")
                for i in range(1, 4):
                    horizon_data = next_pred[f'horizon_{i}']
                    change_pct = horizon_data['change'] * 100
                    direction = "📈" if horizon_data['change'] > 0 else "📉"
                    print(f"     {i} ngày tới: {horizon_data['price']:.2f} ({change_pct:+.2f}%) {direction}")


    def save_models(self, out_dir="saved_models"):
        os.makedirs(out_dir, exist_ok=True)
        for symbol, models in self.models.items():
            for target_name, model in models.items():
                model_dict = {
                    "weights": model["weights"],
                    "feature_names": model["feature_names"],
                    "feature_stats": model["feature_stats"]
                }
                fname = f"{out_dir}/{symbol}_{target_name}_model.json"
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(model_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu mô hình vào thư mục {out_dir}")

    def load_models(self, in_dir="saved_models"):
        self.models = {}
        for symbol in self.symbols:
            self.models[symbol] = {}
            for target_name in ['target_1', 'target_2', 'target_3']:
                fname = f"{in_dir}/{symbol}_{target_name}_model.json"
                if os.path.exists(fname):
                    with open(fname, "r", encoding="utf-8") as f:
                        model_dict = json.load(f)
                    self.models[symbol][target_name] = model_dict
        print(f"✅ Đã nạp mô hình từ thư mục {in_dir}")


def main():
    print("🚀 KHỞI ĐỘNG MÔ HÌNH DỰ BÁO NGẮN HẠN TỐI ƯU")
    print("=" * 80)
    predictor = OptimizedShortTermPredictor(['VCB', 'VNM', 'FPT'])
    predictor.load_data()
    predictor.train_all_models()
    predictor.save_models()  # Lưu mô hình sau khi huấn luyện
    predictor.print_summary()
    predictor.create_latex_plots()
    predictor.create_latex_tikz_code()
    predictor.save_comprehensive_results()
    print("\n✅ HOÀN THÀNH!")
    print("📊 Kiểm tra thư mục 'model_figuress' để xem:")
    print("   - Các file CSV cho biểu đồ")
    print("   - File prediction_plots.tex cho LaTeX")
    print("   - File comprehensive_results.json")
    print("   - Thư mục saved_models chứa file mô hình")
    return predictor


if __name__ == "__main__":
    predictor = main()
