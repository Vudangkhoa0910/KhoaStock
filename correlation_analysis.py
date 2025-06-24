import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Read data files
data_path = 'data/collected_data/daily'
symbols = ['VNINDEX', 'VN30F1M', 'FPT', 'VNM', 'VCB']
dfs = {}

for symbol in symbols:
    file_path = f'{data_path}/{symbol}_daily.csv'
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    dfs[symbol] = df

# Create combined DataFrame for correlation analysis
combined_data = pd.DataFrame()
for symbol in symbols:
    combined_data[f'{symbol}_close'] = dfs[symbol]['close']
    combined_data[f'{symbol}_volume'] = np.log(dfs[symbol]['volume'])  # Log transform volume

# 1. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = combined_data.corr(method='pearson')
mask = np.triu(np.ones_like(correlation_matrix), k=0)
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .5})

plt.title('Correlation Heatmap of Stock Prices and Volumes')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Cross-correlation Plot
def plot_ccf(y1, y2, nlags=20):
    ccf_coef = ccf(y1, y2, adjusted=False)[:nlags+1]
    
    plt.figure(figsize=(12, 6))
    plt.stem(range(len(ccf_coef)), ccf_coef, use_line_collection=True)
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=-0.2, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Lag (days)')
    plt.ylabel('Cross-correlation coefficient')
    plt.title('Cross-correlation between FPT and VN-Index Close Prices')
    plt.grid(True, alpha=0.3)
    return plt

# Calculate and plot CCF
fpt_returns = dfs['FPT']['close'].pct_change().dropna()
vnindex_returns = dfs['VNINDEX']['close'].pct_change().dropna()
plot_ccf(fpt_returns, vnindex_returns)
plt.savefig('figures/cross_correlation_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Correlation analysis plots have been generated in the 'figures' directory.") 