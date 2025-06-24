import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory if not exists
os.makedirs("data/processed_data_2/analysis", exist_ok=True)

def load_data(symbol):
    # Load price data
    price_df = pd.read_csv(f"data/collected_data/daily/{symbol}_daily.csv")
    price_df['time'] = pd.to_datetime(price_df['time'])
    price_df.set_index('time', inplace=True)
    
    # Load technical indicators
    tech_df = pd.read_csv(f"data/collected_data/technical/{symbol}_technical.csv")
    tech_df['time'] = pd.to_datetime(tech_df['time'])
    tech_df.set_index('time', inplace=True)
    
    # Select required columns
    price_df = price_df[['close', 'volume']]
    tech_df = tech_df[['RSI', 'Force_Index']]  # Using Force_Index instead of MFI
    
    # Merge data
    df = pd.merge(price_df, tech_df, left_index=True, right_index=True, how='inner')
    df.columns = ['Close', 'Volume', 'RSI', 'MFI']  # Rename Force_Index to MFI for consistency
    return df

def plot_correlation_matrix(symbols=['FPT', 'VNM', 'VCB']):
    plt.figure(figsize=(15, 5))
    
    for i, symbol in enumerate(symbols, 1):
        df = load_data(symbol)
        features = ['Close', 'RSI', 'MFI', 'Volume']
        corr = df[features].corr(method='pearson')
        
        plt.subplot(1, 3, i)
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'{symbol} Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('data/processed_data_2/analysis/correlation_matrix.png')
    plt.close()

def plot_autocorrelation(symbols=['FPT', 'VNM', 'VCB'], lags=20):
    plt.figure(figsize=(15, 5))
    
    for i, symbol in enumerate(symbols, 1):
        df = load_data(symbol)
        acf_values = acf(df['Close'], nlags=lags)
        
        plt.subplot(1, 3, i)
        plt.bar(range(lags + 1), acf_values)
        plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
        plt.axhline(y=1.96/np.sqrt(len(df)), linestyle='--', color='gray', alpha=0.3)
        plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle='--', color='gray', alpha=0.3)
        plt.title(f'{symbol} Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
    
    plt.tight_layout()
    plt.savefig('data/processed_data_2/analysis/autocorrelation.png')
    plt.close()

def plot_granger_causality(symbols=['FPT', 'VNM', 'VCB']):
    plt.figure(figsize=(15, 5))
    maxlag = 5
    
    for i, symbol in enumerate(symbols, 1):
        df = load_data(symbol)
        gc_test = grangercausalitytests(df[['Close', 'Volume']], maxlag=maxlag, verbose=False)
        
        # Extract p-values
        p_values = [gc_test[lag+1][0]['ssr_chi2test'][1] for lag in range(maxlag)]
        
        plt.subplot(1, 3, i)
        plt.plot(range(1, maxlag+1), p_values, marker='o')
        plt.axhline(y=0.05, linestyle='--', color='red', alpha=0.3)
        plt.title(f'{symbol} Granger Causality Test\nVolume → Close')
        plt.xlabel('Lag')
        plt.ylabel('p-value')
    
    plt.tight_layout()
    plt.savefig('data/processed_data_2/analysis/granger_causality.png')
    plt.close()

def plot_adf_test_results(symbols=['FPT', 'VNM', 'VCB']):
    plt.figure(figsize=(15, 5))
    
    for i, symbol in enumerate(symbols, 1):
        df = load_data(symbol)
        
        # Original series
        adf_orig = adfuller(df['Close'])
        # First difference
        adf_diff = adfuller(df['Close'].diff().dropna())
        
        plt.subplot(1, 3, i)
        plt.bar(['Original', 'First Difference'], 
                [-adf_orig[0], -adf_diff[0]],
                color=['blue', 'green'])
        plt.axhline(y=-adf_orig[4]['5%'], linestyle='--', color='red', alpha=0.3)
        plt.title(f'{symbol} ADF Test Statistics')
        plt.ylabel('Negative ADF Statistic')
    
    plt.tight_layout()
    plt.savefig('data/processed_data_2/analysis/adf_test.png')
    plt.close()

if __name__ == "__main__":
    # Generate all plots
    plot_correlation_matrix()
    plot_autocorrelation()
    plot_granger_causality()
    plot_adf_test_results() 