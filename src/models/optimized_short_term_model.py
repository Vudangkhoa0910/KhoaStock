# Author: Vu Dang Khoa
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
import math
import json
import csv
import os

class StockPredictor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['VCB', 'VNM', 'FPT']
        self.data = {}
        self.models = {}
        self.predictions = {}
        self.results = {}

    def read_csv(self, path):
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        data.append({
                            'datetime': self._parse_date(row['time']),
                            'Open': float(row['Open']),
                            'High': float(row['High']),
                            'Low': float(row['Low']),
                            'Close': float(row['Close']),
                            'Volume': float(row['Volume'])
                        })
                    except (ValueError, KeyError):
                        continue
            return sorted(data, key=lambda x: x['datetime'])
        except Exception as e:
            print(f"Failed to read {path}: {str(e)}")
            return []

    def _parse_date(self, date_str):
        try:
            return datetime.strptime(date_str.split()[0], '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except:
                return None

    def load_data(self, base_path=None):
        if not base_path:
            base_path = Path.home() / "Working/KhoaStock/data/collected_data"
            
        for sym in self.symbols:
            try:
                daily = list(Path(base_path).glob(f"daily/{sym}_daily*.csv"))[-1]
                self.data[sym] = self.read_csv(daily)
            except Exception as e:
                print(f"Failed to load {sym}: {e}")

    def calc_momentum(self, prices, volumes):
        features = {}
        if len(prices) < 5:
            return features

        features.update({
            'price_mom_1': (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0,
            'price_mom_2': (prices[-1] - prices[-3]) / prices[-3] if len(prices) > 2 and prices[-3] != 0 else 0,
            'price_mom_3': (prices[-1] - prices[-4]) / prices[-4] if len(prices) > 3 and prices[-4] != 0 else 0
        })

        if len(volumes) >= 3:
            features.update({
                'vol_mom_1': (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0,
                'vol_mom_2': (volumes[-1] - volumes[-3]) / volumes[-3] if volumes[-3] != 0 else 0
            })

        recent_prices = prices[-5:]
        if len(recent_prices) > 1:
            mean = sum(recent_prices) / len(recent_prices)
            var = sum((p - mean)**2 for p in recent_prices) / len(recent_prices)
            features['price_vol'] = math.sqrt(var) / mean if mean != 0 else 0

        if len(volumes) >= 5:
            recent_vols = volumes[-5:]
            mean_vol = sum(recent_vols[:-1]) / len(recent_vols[:-1])
            features['vol_surge'] = volumes[-1] / mean_vol if mean_vol != 0 else 1

        return features

    def calc_technicals(self, data):
        closes = [row['Close'] for row in data]
        highs = [row['High'] for row in data]
        lows = [row['Low'] for row in data]
        volumes = [row['Volume'] for row in data]
        
        indicators = []
        for i in range(len(data)):
            if i < 5:
                indicators.append({})
                continue

            curr_data = {
                'close': closes[i],
                'high': highs[i],
                'low': lows[i],
                'open': data[i]['Open'],
                'volume': volumes[i]
            }

            total_range = curr_data['high'] - curr_data['low']
            body_size = abs(curr_data['close'] - curr_data['open'])

            ind = {
                'body_ratio': body_size / total_range if total_range != 0 else 0
            }

            if curr_data['close'] > curr_data['open']:
                upper = curr_data['high'] - curr_data['close']
                lower = curr_data['open'] - curr_data['low']
            else:
                upper = curr_data['high'] - curr_data['open']
                lower = curr_data['close'] - curr_data['low']

            ind.update({
                'upper_shadow': upper / total_range if total_range != 0 else 0,
                'lower_shadow': lower / total_range if total_range != 0 else 0
            })

            recent_high = max(highs[i-4:i+1])
            recent_low = min(lows[i-4:i+1])
            price_range = recent_high - recent_low
            if price_range != 0:
                ind['price_position'] = (closes[i] - recent_low) / price_range

            vol_avg = sum(volumes[i-4:i]) / 4
            ind['vol_ratio'] = volumes[i] / vol_avg if vol_avg != 0 else 1

            if i >= 2:
                prev_change = closes[i-1] - closes[i-2]
                curr_change = closes[i] - closes[i-1]
                ind['price_accel'] = curr_change - prev_change

            momentum = self.calc_momentum(closes[:i+1], volumes[:i+1])
            ind.update(momentum)
            indicators.append(ind)

        return indicators

    def prepare_features(self, data):
        if not data:
            return []

        technicals = self.calc_technicals(data)
        closes = [row['Close'] for row in data]
        features = []

        for i in range(len(data)):
            if i < 10:
                continue

            targets = {}
            for h in range(1, 4):
                if i + h < len(closes):
                    targets[f'target_{h}'] = closes[i+h] / closes[i] - 1

            feat = technicals[i].copy()
            
            for lag in range(1, 4):
                idx = i - lag
                if idx >= 0:
                    feat[f'close_lag_{lag}'] = closes[idx]
                    feat[f'tech_lag_{lag}'] = technicals[idx]

            feat.update({
                'weekday': data[i]['datetime'].weekday(),
                'hour': data[i]['datetime'].hour,
                'price': closes[i],
                'datetime': data[i]['datetime']
            })

            feat.update(targets)
            features.append(feat)

        return features

    def normalize(self, data, means=None, stds=None):
        if not means or not stds:
            means = {}
            stds = {}
            for key in data[0].keys():
                if isinstance(data[0][key], (int, float)):
                    values = [x[key] for x in data]
                    means[key] = sum(values) / len(values)
                    var = sum((x - means[key])**2 for x in values) / len(values)
                    stds[key] = math.sqrt(var) if var > 0 else 1

        normalized = []
        for row in data:
            norm_row = {}
            for key, val in row.items():
                if key in means:
                    norm_row[key] = (val - means[key]) / stds[key]
                else:
                    norm_row[key] = val
            normalized.append(norm_row)

        return normalized, means, stds

    def train(self, sym):
        print(f"Training model for {sym}")
        
        if sym not in self.data:
            return False
            
        features = self.prepare_features(self.data[sym])
        if len(features) < 30:
            return False
            
        features = features[:-3]
        
        exclude = {'target_1', 'target_2', 'target_3', 'price', 'datetime'}
        feature_cols = [k for k in features[0].keys() 
                       if k not in exclude and features[0][k] is not None]
                       
        split = int(len(features) * 0.85)
        train_data = features[:split]
        test_data = features[split:]
        
        train_norm, means, stds = self.normalize(train_data)
        test_norm, _, _ = self.normalize(test_data, means, stds)
        
        models = {}
        results = {}
        
        for target in ['target_1', 'target_2', 'target_3']:
            X_train = [[row[col] for col in feature_cols] for row in train_norm]
            y_train = [row[target] for row in train_norm if target in row]
            
            X_test = [[row[col] for col in feature_cols] for row in test_norm]
            y_test = [row[target] for row in test_norm if target in row]
            
            model = self._train_model(X_train, y_train)
            y_pred = self._predict(X_test, model)
            
            models[target] = model
            results[target] = self._evaluate(y_test, y_pred)
            
        self.models[sym] = {
            'models': models,
            'features': feature_cols,
            'means': means,
            'stds': stds
        }
        
        self.predictions[sym] = results
        return True

    def _train_model(self, X, y):
        n_features = len(X[0])
        weights = np.zeros(n_features + 1)
        X = np.column_stack([np.ones(len(X)), X])
        
        alpha = 0.01
        n_iter = 1000
        lr = 0.01
        
        for _ in range(n_iter):
            y_pred = np.dot(X, weights)
            error = y_pred - y
            gradient = np.dot(X.T, error) / len(X)
            weights -= lr * (gradient + alpha * weights)
            
        return weights

    def _predict(self, X, weights):
        X = np.column_stack([np.ones(len(X)), X])
        return np.dot(X, weights)

    def _evaluate(self, y_true, y_pred):
        mse = sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)
        mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
        
        mean_y = sum(y_true) / len(y_true)
        ss_tot = sum((y - mean_y)**2 for y in y_true)
        ss_res = sum((t - p)**2 for t, p in zip(y_true, y_pred))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {'mse': mse, 'mae': mae, 'r2': r2}

    def train_all(self):
        for sym in self.symbols:
            self.train(sym)

    def predict(self, sym):
        if sym not in self.models:
            return None

        features = self.prepare_features(self.data[sym])
        if not features:
            return None

        latest = features[-1]
        model_data = self.models[sym]
        
        feature_vector = [latest[f] for f in model_data['features']]
        feature_vector = [(f - model_data['means'].get(f, 0)) / 
                         model_data['stds'].get(f, 1) 
                         for f in feature_vector]
        
        predictions = {}
        for horizon, model in model_data['models'].items():
            pred = self._predict([feature_vector], model)[0]
            predictions[horizon] = {
                'change': pred,
                'price': latest['price'] * (1 + pred)
            }
            
        predictions['current_price'] = latest['price']
        return predictions

    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True)
        for sym in self.symbols:
            if sym in self.models:
                with open(f"{path}/{sym}_model.json", 'w') as f:
                    json.dump(self.models[sym], f)

    def load_models(self, path="models"):
        for sym in self.symbols:
            try:
                with open(f"{path}/{sym}_model.json") as f:
                    self.models[sym] = json.load(f)
            except:
                print(f"No saved model for {sym}")

def main():
    print("Starting Stock Prediction Model")
    print("=" * 50)
    
    model = StockPredictor(['VCB', 'VNM', 'FPT'])
    model.load_data()
    model.train_all()
    model.save_models()
    
    for sym in model.symbols:
        pred = model.predict(sym)
        if pred:
            print(f"\n{sym} Predictions:")
            print(f"Current: {pred['current_price']:.2f}")
            for days in range(1, 4):
                key = f'target_{days}'
                if key in pred:
                    print(f"{days}-day: {pred[key]['price']:.2f} ({pred[key]['change']:.2%})")
    
    return model

if __name__ == "__main__":
    model = main()
