import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob

# Set style for better visualization
plt.style.use('seaborn')
sns.set_palette("husl")

def analyze_daily_data():
    """Analyze daily trading data"""
    data_path = 'data/collected_data/daily'
    symbols = ['VNINDEX', 'VN30F1M', 'FPT', 'VNM', 'VCB']
    all_data = {}
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures/data_analysis', exist_ok=True)
    
    # Read all daily data
    for symbol in symbols:
        file_path = f'{data_path}/{symbol}_daily.csv'
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        all_data[symbol] = df

    # 1. Price Movement Analysis
    plt.figure(figsize=(15, 8))
    for symbol in symbols:
        plt.plot(all_data[symbol]['time'], 
                all_data[symbol]['close'], 
                label=symbol,
                linewidth=2)
    plt.title('Price Movement Comparison (2024-2025)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/data_analysis/price_movement.png', dpi=300)
    plt.close()

    # 2. Volume Distribution
    plt.figure(figsize=(15, 8))
    for symbol in symbols:
        if symbol != 'VNINDEX':  # Exclude VNINDEX as it's an index
            sns.kdeplot(data=np.log10(all_data[symbol]['volume']), 
                       label=symbol)
    plt.title('Trading Volume Distribution (Log10 Scale)')
    plt.xlabel('Log10(Volume)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/data_analysis/volume_distribution.png', dpi=300)
    plt.close()

    # 3. Daily Returns Box Plot
    returns_data = pd.DataFrame()
    for symbol in symbols:
        returns_data[symbol] = all_data[symbol]['close'].pct_change().dropna() * 100

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=returns_data)
    plt.title('Daily Returns Distribution (%)')
    plt.ylabel('Daily Return (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/data_analysis/returns_distribution.png', dpi=300)
    plt.close()

    # 4. Data Statistics Summary
    stats_summary = pd.DataFrame()
    for symbol in symbols:
        data = all_data[symbol]
        stats = {
            'Start Date': data['time'].min().strftime('%Y-%m-%d'),
            'End Date': data['time'].max().strftime('%Y-%m-%d'),
            'Trading Days': len(data),
            'Avg Price': data['close'].mean(),
            'Min Price': data['close'].min(),
            'Max Price': data['close'].max(),
            'Price Std Dev': data['close'].std(),
            'Avg Daily Return %': data['close'].pct_change().mean() * 100,
            'Return Std Dev %': data['close'].pct_change().std() * 100
        }
        if symbol != 'VNINDEX':
            stats.update({
                'Avg Volume': data['volume'].mean(),
                'Min Volume': data['volume'].min(),
                'Max Volume': data['volume'].max()
            })
        stats_summary[symbol] = pd.Series(stats)

    # Save statistics to CSV
    stats_summary.to_csv('figures/data_analysis/data_statistics.csv')

    # 5. Create a summary visualization
    plt.figure(figsize=(15, 10))
    
    # Plot normalized prices for comparison
    for symbol in symbols:
        normalized_price = all_data[symbol]['close'] / all_data[symbol]['close'].iloc[0] * 100
        plt.plot(all_data[symbol]['time'], 
                normalized_price,
                label=f'{symbol} (Base=100)',
                linewidth=2)
    
    plt.title('Normalized Price Performance Comparison (Base=100)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/data_analysis/normalized_performance.png', dpi=300)
    plt.close()

    return stats_summary

def print_data_collection_summary():
    """Print summary of collected data structure"""
    base_path = 'data/collected_data'
    summary = {
        'daily': len(glob.glob(f'{base_path}/daily/*.csv')),
        'intraday': len(glob.glob(f'{base_path}/intraday/*.csv')),
        'fundamental': len(glob.glob(f'{base_path}/fundamental/*.csv')),
        'technical': len(glob.glob(f'{base_path}/technical/*.csv')),
        'market_data': len(glob.glob(f'{base_path}/market_data/*.csv')),
        'news': len(glob.glob(f'{base_path}/news/*.csv')),
        'company': len(glob.glob(f'{base_path}/company/*.csv')),
        'trading_stats': len(glob.glob(f'{base_path}/trading_stats/*.csv'))
    }
    
    # Create bar plot of data collection summary
    plt.figure(figsize=(12, 6))
    plt.bar(summary.keys(), summary.values())
    plt.title('Number of Files by Data Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/data_analysis/data_collection_summary.png', dpi=300)
    plt.close()
    
    # Save summary to CSV
    pd.Series(summary).to_csv('figures/data_analysis/data_collection_summary.csv')
    
    return summary

if __name__ == "__main__":
    print("Analyzing collected data...")
    stats_summary = analyze_daily_data()
    data_collection_summary = print_data_collection_summary()
    
    print("\nAnalysis complete! Results saved in figures/data_analysis/")
    print("\nData Collection Summary:")
    for category, count in data_collection_summary.items():
        print(f"{category}: {count} files") 