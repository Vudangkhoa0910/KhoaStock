# Author: Vu Dang Khoa
import csv
import json
import os
import math
import statistics
from datetime import datetime
from collections import defaultdict

class StockPredictor:
    def __init__(self, symbols=['VCB', 'VNM', 'FPT']):
        self.symbols = symbols
        self.data = {}
        self.models = {}
        self.predictions = {}
        
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
            print(f"Error reading {filepath}: {str(e)}")
            return []
    
    def parse_date(self, date_str):
        try:
            return datetime.strptime(date_str.split()[0], '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                return datetime.now()
    
    def load_data(self, data_path="/Users/vudangkhoa/Working/KhoaStock/data/collected_data"):
        print("Loading data...")
        
        for symbol in self.symbols:
            try:
                tech_file = f"{data_path}/technical/{symbol}_technical.csv"
                if os.path.exists(tech_file):
                    tech_data = self.load_csv_data(tech_file)
                    
                    for row in tech_data:
                        row['datetime'] = self.parse_date(row['time'])
                    
                    tech_data.sort(key=lambda x: x['datetime'])
                    self.data[symbol] = tech_data
                    print(f"Loaded {len(tech_data)} rows for {symbol}")
                else:
                    print(f"File not found: {tech_file}")
                    
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    def calculate_moving_average(self, values, window):
        result = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(None)
            else:
                avg = sum(values[i-window+1:i+1]) / window
                result.append(avg)
        return result
    
    def calculate_percentage_change(self, values, period=1):
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
        if not data:
            return []
        
        close_prices = [row['Close'] for row in data if row['Close'] is not None]
        volumes = [row['Volume'] for row in data if row['Volume'] is not None]
        highs = [row['High'] for row in data if row['High'] is not None]
        lows = [row['Low'] for row in data if row['Low'] is not None]
        opens = [row['Open'] for row in data if row['Open'] is not None]
        
        price_change_1d = self.calculate_percentage_change(close_prices, 1)
        price_change_3d = self.calculate_percentage_change(close_prices, 3)
        price_change_5d = self.calculate_percentage_change(close_prices, 5)
        volume_change = self.calculate_percentage_change(volumes, 1)
        
        ma5 = self.calculate_moving_average(close_prices, 5)
        ma10 = self.calculate_moving_average(close_prices, 10)
        ma20 = self.calculate_moving_average(close_prices, 20)
        
        rsi_values = self.calculate_rsi(close_prices)
        
        features_data = []
        for i in range(len(data)):
            if i < 20:
                continue
                
            row = data[i]
            
            hl_ratio = (highs[i] - lows[i]) / close_prices[i] if close_prices[i] != 0 else 0
            oc_ratio = (close_prices[i] - opens[i]) / opens[i] if opens[i] != 0 else 0
            
            price_ma5_ratio = close_prices[i] / ma5[i] if ma5[i] is not None and ma5[i] != 0 else 1
            price_ma10_ratio = close_prices[i] / ma10[i] if ma10[i] is not None and ma10[i] != 0 else 1
            price_ma20_ratio = close_prices[i] / ma20[i] if ma20[i] is not None and ma20[i] != 0 else 1
            
            close_lag1 = close_prices[i-1] if i > 0 else close_prices[i]
            close_lag2 = close_prices[i-2] if i > 1 else close_prices[i]
            volume_lag1 = volumes[i-1] if i > 0 else volumes[i]
            
            day_of_week = row['datetime'].weekday()
            month = row['datetime'].month
            
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
        normalized_data = []
        stats = {}
        
        for feature in feature_names:
            values = [row[feature] for row in data if feature in row]
            if len(values) > 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                std_val = max(std_val, 1e-8)
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
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    def matrix_inverse_2x2(self, matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        det = a*d - b*c
        if abs(det) < 1e-10:
            return [[1, 0], [0, 1]]
        return [[d/det, -b/det], [-c/det, a/det]]
    
    def simple_linear_regression(self, X, y):
        n = len(X)
        if n == 0:
            return [0, 0]
        
        X_with_bias = [[1] + row for row in X]
        
        if len(X[0]) == 1:
            sum_x = sum(row[0] for row in X)
            sum_y = sum(y)
            sum_xy = sum(X[i][0] * y[i] for i in range(n))
            sum_x2 = sum(row[0]**2 for row in X)
            
            if n * sum_x2 - sum_x**2 != 0:
                a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
                b = (sum_y - a * sum_x) / n
                return [b, a]
            else:
                return [sum_y/n, 0]
        
        mean_y = sum(y) / len(y)
        
        weights = [mean_y]
        for j in range(len(X[0])):
            feature_values = [row[j] for row in X]
            if len(set(feature_values)) > 1:
                mean_x = sum(feature_values) / len(feature_values)
                numerator = sum((feature_values[i] - mean_x) * (y[i] - mean_y) for i in range(len(y)))
                denominator = sum((x - mean_x)**2 for x in feature_values)
                if denominator != 0:
                    weight = numerator / denominator * 0.1
                else:
                    weight = 0
            else:
                weight = 0
            weights.append(weight)
        
        return weights
    
    def predict(self, X, weights):
        predictions = []
        for row in X:
            pred = weights[0]
            for i, feature in enumerate(row):
                if i + 1 < len(weights):
                    pred += feature * weights[i + 1]
            predictions.append(pred)
        return predictions
    
    def train_model(self, symbol):
        print(f"\nTraining model for {symbol}...")
        
        if symbol not in self.data:
            print(f"No data for {symbol}")
            return False
        
        features_data = self.create_features(self.data[symbol])
        
        if len(features_data) < 30:
            print(f"Not enough data to train {symbol}")
            return False
        
        features_data = features_data[:-1]
        
        split_idx = int(len(features_data) * 0.8)
        train_data = features_data[:split_idx]
        test_data = features_data[split_idx:]
        
        feature_names = [
            'price_change_1d', 'price_change_3d', 'price_change_5d',
            'volume_change', 'hl_ratio', 'oc_ratio',
            'price_ma5_ratio', 'price_ma10_ratio', 'price_ma20_ratio',
            'rsi', 'day_of_week', 'month'
        ]
        
        train_normalized, stats = self.normalize_data(train_data, feature_names)
        
        X_train = []
        y_train = []
        for row in train_normalized:
            features = [row[f] for f in feature_names]
            X_train.append(features)
            y_train.append(row['target'])
        
        weights = self.simple_linear_regression(X_train, y_train)
        
        test_normalized = []
        for row in test_data:
            normalized_row = row.copy()
            for feature in feature_names:
                if feature in stats:
                    normalized_row[feature] = (row[feature] - stats[feature]['mean']) / stats[feature]['std']
            test_normalized.append(normalized_row)
        
        X_test = []
        y_test = []
        for row in test_normalized:
            features = [row[f] for f in feature_names]
            X_test.append(features)
            y_test.append(row['target'])
        
        y_pred = self.predict(X_test, weights)
        
        mse = sum((y_test[i] - y_pred[i])**2 for i in range(len(y_test))) / len(y_test)
        mae = sum(abs(y_test[i] - y_pred[i]) for i in range(len(y_test))) / len(y_test)
        
        mean_y = sum(y_test) / len(y_test)
        ss_res = sum((y_test[i] - y_pred[i])**2 for i in range(len(y_test)))
        ss_tot = sum((y - mean_y)**2 for y in y_test)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.models[symbol] = {
            'weights': weights,
            'feature_names': feature_names,
            'stats': stats,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        self.predictions[symbol] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'test_data': test_data,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"Training completed for {symbol}")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        
        return True
    
    def train_all_models(self):
        print("Start training all models...")
        
        for symbol in self.symbols:
            self.train_model(symbol)
    
    def predict_next_day(self, symbol):
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]
        features_data = self.create_features(self.data[symbol])
        
        if not features_data:
            return None
        
        latest_data = features_data[-1]
        
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
        
        prediction = self.predict([normalized_features], model['weights'])[0]
        current_price = latest_data['current_price']
        predicted_price = current_price * (1 + prediction)
        
        return {
            'current_price': current_price,
            'predicted_change': prediction,
            'predicted_price': predicted_price
        }
    
    def save_results(self, filename="prediction_results.json"):
        results = {}
        
        for symbol in self.symbols:
            if symbol in self.models and symbol in self.predictions:
                pred_data = self.predictions[symbol]
                model_data = self.models[symbol]
                
                y_true_direction = [1 if y > 0 else -1 for y in pred_data['y_true']]
                y_pred_direction = [1 if y > 0 else -1 for y in pred_data['y_pred']]
                direction_accuracy = sum(1 for i in range(len(y_true_direction)) 
                                       if y_true_direction[i] == y_pred_direction[i]) / len(y_true_direction) * 100
                
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
        
        print(f"Results saved to {filename}")
    
    def create_simple_plot_data(self):
        for symbol, pred_data in self.predictions.items():
            filename = f"{symbol}_prediction_plot.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Index', 'True_Value', 'Predicted_Value', 'Error'])
                
                for i in range(len(pred_data['y_true'])):
                    true_val = pred_data['y_true'][i]
                    pred_val = pred_data['y_pred'][i]
                    error = pred_val - true_val
                    writer.writerow([i, true_val, pred_val, error])
            
            print(f"Plot data created for {symbol} in {filename}")
    
    def print_summary(self):
        print("\nMODEL PREDICTION SUMMARY")
        print("="*60)
        
        if not self.predictions:
            print("No prediction results")
            return
        
        for symbol, pred_data in self.predictions.items():
            print(f"\nResults for {symbol}:")
            print(f"MSE: {pred_data['mse']:.6f}")
            print(f"MAE: {pred_data['mae']:.6f}")
            print(f"R²: {pred_data['r2']:.4f}")
            
            y_true_direction = [1 if y > 0 else -1 for y in pred_data['y_true']]
            y_pred_direction = [1 if y > 0 else -1 for y in pred_data['y_pred']]
            direction_accuracy = sum(1 for i in range(len(y_true_direction)) 
                                   if y_true_direction[i] == y_pred_direction[i]) / len(y_true_direction) * 100
            print(f"Direction accuracy: {direction_accuracy:.1f}%")
            
            next_pred = self.predict_next_day(symbol)
            if next_pred:
                print(f"Current price: {next_pred['current_price']:.2f}")
                print(f"Predicted change: {next_pred['predicted_change']:.4f} ({next_pred['predicted_change']*100:.2f}%)")
                print(f"Predicted price: {next_pred['predicted_price']:.2f}")


def main():
    print("Starting Stock Prediction Model")
    print("=" * 60)
    
    predictor = StockPredictor(['VCB', 'VNM', 'FPT'])
    
    predictor.load_data()
    predictor.train_all_models()
    predictor.print_summary()
    predictor.save_results()
    predictor.create_simple_plot_data()
    
    print("\nProcess completed! Check the generated JSON and CSV files.")
    print("Use Excel or Google Sheets to visualize the CSV data.")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
