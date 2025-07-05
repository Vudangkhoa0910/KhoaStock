# Author: Vu Dang Khoa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode()

class ModelManager:
    def __init__(self, symbols=['VCB', 'VNM', 'FPT']):
        self.symbols = symbols
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.predictions = {}
        self.feature_importance = {}
        
    def load_data(self, data_path="/Users/vudangkhoa/Working/KhoaStock/data/collected_data"):
        print("Loading market data...")
        
        for symbol in self.symbols:
            try:
                tech_file = f"{data_path}/technical/{symbol}_technical.csv"
                tech_data = pd.read_csv(tech_file)
                tech_data['time'] = pd.to_datetime(tech_data['time'])
                tech_data = tech_data.sort_values('time').reset_index(drop=True)
                
                daily_file = f"{data_path}/daily/{symbol}_daily.csv"
                daily_data = pd.read_csv(daily_file)
                daily_data['time'] = pd.to_datetime(daily_data['time'])
                daily_data = daily_data.sort_values('time').reset_index(drop=True)
                
                merged_data = pd.merge(daily_data, tech_data, on='time', how='left', suffixes=('', '_tech'))
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in merged_data.columns and f'{col}_tech' in merged_data.columns:
                        merged_data[col] = merged_data[f'{col}_tech'].fillna(merged_data[col])
                
                cols_to_drop = [col for col in merged_data.columns if col.endswith('_tech')]
                merged_data = merged_data.drop(columns=cols_to_drop)
                
                self.data[symbol] = merged_data
                print(f"Loaded {len(merged_data)} rows for {symbol}")
                
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    def create_features(self, df):
        features_df = df.copy()
        
        features_df['price_change'] = features_df['Close'].pct_change()
        features_df['price_change_1d'] = features_df['Close'].pct_change(1)
        features_df['price_change_3d'] = features_df['Close'].pct_change(3)
        features_df['price_change_5d'] = features_df['Close'].pct_change(5)
        
        features_df['volume_change'] = features_df['Volume'].pct_change()
        features_df['volume_ma5'] = features_df['Volume'].rolling(window=5).mean()
        features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma5']
        
        features_df['hl_ratio'] = (features_df['High'] - features_df['Low']) / features_df['Close']
        features_df['oc_ratio'] = (features_df['Close'] - features_df['Open']) / features_df['Open']
        
        for window in [5, 10, 20]:
            features_df[f'ma_{window}'] = features_df['Close'].rolling(window=window).mean()
            features_df[f'price_ma_{window}_ratio'] = features_df['Close'] / features_df[f'ma_{window}']
        
        if 'RSI' not in features_df.columns:
            delta = features_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['RSI'] = 100 - (100 / (1 + rs))
        
        if all(col in features_df.columns for col in ['BB_High', 'BB_Low', 'BB_Mid']):
            features_df['bb_position'] = (features_df['Close'] - features_df['BB_Low']) / (features_df['BB_High'] - features_df['BB_Low'])
        
        features_df['target_next_close'] = features_df['Close'].shift(-1)
        features_df['target_price_change'] = features_df['target_next_close'] / features_df['Close'] - 1
        
        for lag in [1, 2, 3]:
            features_df[f'close_lag_{lag}'] = features_df['Close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['Volume'].shift(lag)
            features_df[f'price_change_lag_{lag}'] = features_df['price_change'].shift(lag)
        
        features_df['day_of_week'] = pd.to_datetime(features_df['time']).dt.dayofweek
        features_df['month'] = pd.to_datetime(features_df['time']).dt.month
        
        return features_df
    
    def select_features(self, df):
        feature_columns = [
            'price_change_1d', 'price_change_3d', 'price_change_5d',
            'volume_change', 'volume_ratio', 
            'hl_ratio', 'oc_ratio',
            'price_ma_5_ratio', 'price_ma_10_ratio', 'price_ma_20_ratio',
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'volume_lag_1', 'price_change_lag_1', 'price_change_lag_2',
            'day_of_week', 'month'
        ]
        
        optional_features = ['RSI', 'MACD', 'bb_position', 'ATR']
        for feat in optional_features:
            if feat in df.columns:
                feature_columns.append(feat)
        
        available_features = [col for col in feature_columns if col in df.columns]
        
        return available_features
    
    def train_model(self, symbol):
        print(f"\nTraining model for {symbol}...")
        
        df = self.data[symbol].copy()
        df = self.create_features(df)
        
        feature_columns = self.select_features(df)
        
        df_clean = df[feature_columns + ['target_price_change']].dropna()
        
        if len(df_clean) < 50:
            print(f"Insufficient data for {symbol}")
            return False
        
        X = df_clean[feature_columns]
        y = df_clean['target_price_change']
        
        split_idx = int(len(df_clean) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models_to_try = {
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
            'Ridge': Ridge(alpha=1.0),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_model_name = None
        
        for name, model in models_to_try.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            print(f"  {name}: MSE = {mse:.6f}")
            
            if mse < best_score:
                best_score = mse
                best_model = model
                best_model_name = name
        
        self.models[symbol] = best_model
        self.scalers[symbol] = scaler
        
        y_pred_best = best_model.predict(X_test_scaled)
        
        self.predictions[symbol] = {
            'y_true': y_test.values,
            'y_pred': y_pred_best,
            'dates': df_clean.iloc[split_idx:]['time'].values if 'time' in df_clean.columns else np.arange(len(y_test)),
            'model_name': best_model_name,
            'feature_names': feature_columns,
            'test_mse': best_score,
            'test_mae': mean_absolute_error(y_test, y_pred_best),
            'test_r2': r2_score(y_test, y_pred_best)
        }
        
        if hasattr(best_model, 'feature_importances_'):
            self.feature_importance[symbol] = dict(zip(feature_columns, best_model.feature_importances_))
        
        print(f"Training completed for {symbol} using {best_model_name}")
        print(f"MSE: {best_score:.6f}, MAE: {self.predictions[symbol]['test_mae']:.6f}, R²: {self.predictions[symbol]['test_r2']:.4f}")
        
        return True
    
    def train_all_models(self):
        print("Starting model training...")
        
        for symbol in self.symbols:
            if symbol in self.data:
                self.train_model(symbol)
            else:
                print(f"No data found for {symbol}")
    
    def predict_next_day(self, symbol, days_ahead=1):
        if symbol not in self.models:
            print(f"No model found for {symbol}")
            return None
        
        df = self.data[symbol].copy()
        df = self.create_features(df)
        
        latest_data = df.iloc[-1:]
        feature_columns = self.predictions[symbol]['feature_names']
        
        X_latest = latest_data[feature_columns]
        
        if X_latest.isnull().any().any():
            print(f"Warning: Missing values in latest data for {symbol}")
            X_latest = X_latest.fillna(method='ffill').fillna(0)
        
        X_latest_scaled = self.scalers[symbol].transform(X_latest)
        
        prediction = self.models[symbol].predict(X_latest_scaled)[0]
        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + prediction)
        
        return {
            'current_price': current_price,
            'predicted_change': prediction,
            'predicted_price': predicted_price,
            'confidence': abs(prediction) if abs(prediction) < 0.1 else 0.1
        }
    
    def save_models(self, save_path="models"):
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for symbol in self.models:
            model_file = f"{save_path}/{symbol}_model.pkl"
            scaler_file = f"{save_path}/{symbol}_scaler.pkl"
            
            joblib.dump(self.models[symbol], model_file)
            joblib.dump(self.scalers[symbol], scaler_file)
            
        print(f"Models saved to {save_path}")
    
    def load_models(self, load_path="models"):
        import os
        
        for symbol in self.symbols:
            model_file = f"{load_path}/{symbol}_model.pkl"
            scaler_file = f"{load_path}/{symbol}_scaler.pkl"
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[symbol] = joblib.load(model_file)
                self.scalers[symbol] = joblib.load(scaler_file)
                print(f"Loaded model for {symbol}")
    
    def plot_prediction_results(self):
        n_symbols = len(self.predictions)
        if n_symbols == 0:
            print("No prediction results to plot")
            return
        
        fig = make_subplots(
            rows=n_symbols, cols=2,
            subplot_titles=[f'{symbol} - Predictions vs Actual' for symbol in self.predictions.keys()] + 
                          [f'{symbol} - Errors' for symbol in self.predictions.keys()],
            vertical_spacing=0.08
        )
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, (symbol, pred_data) in enumerate(self.predictions.items()):
            row = idx + 1
            color = colors[idx % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred_data['y_true']))),
                    y=pred_data['y_true'],
                    name=f'{symbol} Actual',
                    line=dict(color=color),
                    legendgroup=f'group{idx}'
                ),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred_data['y_pred']))),
                    y=pred_data['y_pred'],
                    name=f'{symbol} Predicted',
                    line=dict(color=color, dash='dash'),
                    legendgroup=f'group{idx}'
                ),
                row=row, col=1
            )
            
            errors = pred_data['y_pred'] - pred_data['y_true']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(errors))),
                    y=errors,
                    name=f'{symbol} Error',
                    line=dict(color=color),
                    legendgroup=f'group{idx}',
                    showlegend=False
                ),
                row=row, col=2
            )
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row, col=2)
        
        fig.update_layout(
            height=300 * n_symbols,
            title_text="Model Prediction Results",
            showlegend=True
        )
        
        fig.show()
        fig.write_html("prediction_results.html")
        print("Prediction plots saved to prediction_results.html")
    
    def plot_error_analysis(self):
        if not self.predictions:
            print("No prediction data for analysis")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Error Distribution',
                'Predictions vs Actual', 
                'Performance Over Time',
                'R² Score Distribution'
            ]
        )
        
        all_errors = []
        all_true = []
        all_pred = []
        symbols_data = []
        
        colors = ['blue', 'red', 'green']
        
        for idx, (symbol, pred_data) in enumerate(self.predictions.items()):
            color = colors[idx % len(colors)]
            errors = pred_data['y_pred'] - pred_data['y_true']
            
            all_errors.extend(errors)
            all_true.extend(pred_data['y_true'])
            all_pred.extend(pred_data['y_pred'])
            symbols_data.extend([symbol] * len(errors))
            
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name=f'{symbol}',
                    opacity=0.7,
                    nbinsx=30,
                    legendgroup=f'group{idx}'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_data['y_true'],
                    y=pred_data['y_pred'],
                    mode='markers',
                    name=f'{symbol}',
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=f'group{idx}',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            window_size = min(20, len(pred_data['y_true']) // 4)
            if window_size > 5:
                rolling_r2 = []
                for i in range(window_size, len(pred_data['y_true'])):
                    y_true_window = pred_data['y_true'][i-window_size:i]
                    y_pred_window = pred_data['y_pred'][i-window_size:i]
                    r2 = r2_score(y_true_window, y_pred_window)
                    rolling_r2.append(r2)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(window_size, len(pred_data['y_true']))),
                        y=rolling_r2,
                        name=f'{symbol} R²',
                        line=dict(color=color),
                        legendgroup=f'group{idx}',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        min_val, max_val = min(all_true), max(all_true)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        r2_scores = [pred_data['test_r2'] for pred_data in self.predictions.values()]
        symbols_list = list(self.predictions.keys())
        
        fig.add_trace(
            go.Box(
                y=r2_scores,
                x=symbols_list,
                name='R² Distribution',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Detailed Error Analysis",
            showlegend=True
        )
        
        fig.show()
        fig.write_html("error_analysis.html")
        print("Error analysis plots saved to error_analysis.html")
    
    def print_model_summary(self):
        print("\nMODEL PREDICTION SUMMARY")
        print("="*60)
        
        if not self.predictions:
            print("No prediction results")
            return
        
        for symbol, pred_data in self.predictions.items():
            print(f"\nResults for {symbol}:")
            print(f"Model: {pred_data['model_name']}")
            print(f"MSE: {pred_data['test_mse']:.6f}")
            print(f"MAE: {pred_data['test_mae']:.6f}")
            print(f"R²: {pred_data['test_r2']:.4f}")
            print(f"Test samples: {len(pred_data['y_true'])}")
            
            y_true_direction = np.sign(pred_data['y_true'])
            y_pred_direction = np.sign(pred_data['y_pred'])
            direction_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
            print(f"Direction accuracy: {direction_accuracy:.1f}%")
        
        print("\n" + "="*60)
        
        print("NEXT DAY PREDICTIONS:")
        print("="*60)
        
        for symbol in self.symbols:
            if symbol in self.models:
                prediction = self.predict_next_day(symbol)
                if prediction:
                    print(f"\n{symbol}:")
                    print(f"Current price: {prediction['current_price']:.2f}")
                    print(f"Predicted change: {prediction['predicted_change']:.4f} ({prediction['predicted_change']*100:.2f}%)")
                    print(f"Predicted price: {prediction['predicted_price']:.2f}")


def main():
    print("Starting Lightweight Stock Prediction Model")
    print("=" * 60)
    
    predictor = ModelManager(['VCB', 'VNM', 'FPT'])
    
    predictor.load_data()
    predictor.train_all_models()
    predictor.save_models()
    predictor.print_model_summary()
    
    print("\nGenerating analysis plots...")
    predictor.plot_prediction_results()
    predictor.plot_error_analysis()
    
    print("\nProcess completed! Check the generated HTML files.")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
