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

def analyze_data_category(category_path, category_name):
    """Analyze data within a specific category"""
    files = glob.glob(f'{category_path}/*.csv')
    if not files:
        return None
    
    category_stats = {
        'file_count': len(files),
        'total_rows': 0,
        'date_range': [],
        'columns': set(),
        'data_types': {},
        'sample_stats': {}
    }
    
    for file in files:
        try:
            df = pd.read_csv(file)
            category_stats['total_rows'] += len(df)
            category_stats['columns'].update(df.columns)
            
            # Analyze date ranges if time column exists
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                # Convert to datetime and handle timezone-aware timestamps
                df[time_cols[0]] = pd.to_datetime(df[time_cols[0]]).dt.tz_localize(None)
                category_stats['date_range'].extend([
                    df[time_cols[0]].min(),
                    df[time_cols[0]].max()
                ])
            
            # Analyze numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in category_stats['sample_stats']:
                    category_stats['sample_stats'][col] = {
                        'mean': [],
                        'min': [],
                        'max': [],
                        'std': []
                    }
                stats = category_stats['sample_stats'][col]
                stats['mean'].append(df[col].mean())
                stats['min'].append(df[col].min())
                stats['max'].append(df[col].max())
                stats['std'].append(df[col].std())
            
            # Collect data types
            for col, dtype in df.dtypes.items():
                if col not in category_stats['data_types']:
                    category_stats['data_types'][col] = str(dtype)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Process date range
    if category_stats['date_range']:
        category_stats['date_range'] = [
            min(category_stats['date_range']),
            max(category_stats['date_range'])
        ]
    
    # Calculate average statistics
    for col in category_stats['sample_stats']:
        stats = category_stats['sample_stats'][col]
        stats['mean'] = np.mean(stats['mean'])
        stats['min'] = min(stats['min'])
        stats['max'] = max(stats['max'])
        stats['std'] = np.mean(stats['std'])
    
    return category_stats

def create_data_summary_visualizations(all_stats):
    """Create visualizations for data summary"""
    os.makedirs('figures/data_analysis', exist_ok=True)
    
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('KhoaStock Data Analysis Summary', fontsize=16, y=0.95)
    
    # 1. Data Volume by Category (Top left)
    ax1 = plt.subplot(2, 2, 1)
    categories = []
    rows = []
    for category, stats in all_stats.items():
        if stats:
            categories.append(category)
            rows.append(stats['total_rows'])
    
    bars = ax1.bar(categories, rows)
    ax1.set_title('Total Number of Records by Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Records')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)
    
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Date Range Coverage (Top right)
    ax2 = plt.subplot(2, 2, 2)
    date_ranges = {}
    for category, stats in all_stats.items():
        if stats and stats['date_range']:
            # For news category, limit to most recent year
            if category == 'news':
                end_date = stats['date_range'][1]
                start_date = end_date - pd.DateOffset(years=1)
                date_ranges[category] = {
                    'start': start_date,
                    'end': end_date
                }
            else:
                date_ranges[category] = {
                    'start': stats['date_range'][0],
                    'end': stats['date_range'][1]
                }
    
    if date_ranges:
        categories = list(date_ranges.keys())
        for i, category in enumerate(categories):
            start = date_ranges[category]['start']
            end = date_ranges[category]['end']
            ax2.plot([start, end], [i, i], linewidth=6, label=category)
            
            # Add date labels at the start and end of each line
            ax2.text(start, i, start.strftime('%Y-%m-%d'), 
                    ha='right', va='center', fontsize=8)
            ax2.text(end, i, end.strftime('%Y-%m-%d'), 
                    ha='left', va='center', fontsize=8)
        
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels(categories)
        ax2.set_title('Data Coverage Period by Category')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

    # 3. Column Count by Category (Bottom)
    ax3 = plt.subplot(2, 2, 3)
    categories = []
    column_counts = []
    for category, stats in all_stats.items():
        if stats:
            categories.append(category)
            column_counts.append(len(stats['columns']))
    
    bars = ax3.bar(categories, column_counts)
    ax3.set_title('Number of Columns by Category')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Number of Columns')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Data Types Summary (Bottom right)
    ax4 = plt.subplot(2, 2, 4)
    data_types_summary = {}
    for category, stats in all_stats.items():
        if stats and stats['data_types']:
            type_counts = {}
            for dtype in stats['data_types'].values():
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            data_types_summary[category] = type_counts
    
    # Convert to DataFrame for easier plotting
    dtype_df = pd.DataFrame(data_types_summary).fillna(0)
    dtype_df.plot(kind='bar', stacked=True, ax=ax4)
    ax4.set_title('Data Types Distribution by Category')
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Number of Columns')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Data Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/data_analysis/data_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_data():
    """Analyze all data categories"""
    base_path = 'data/collected_data'
    categories = ['daily', 'intraday', 'fundamental', 'technical', 
                 'market_data', 'news', 'company', 'trading_stats']
    
    all_stats = {}
    for category in categories:
        category_path = f'{base_path}/{category}'
        print(f"\nAnalyzing {category} data...")
        stats = analyze_data_category(category_path, category)
        if stats:
            all_stats[category] = stats
            
            print(f"  - Total records: {stats['total_rows']:,}")
            if stats['date_range']:
                print(f"  - Date range: {stats['date_range'][0].strftime('%Y-%m-%d')} to {stats['date_range'][1].strftime('%Y-%m-%d')}")
            print(f"  - Number of columns: {len(stats['columns'])}")
            print(f"  - Columns: {', '.join(sorted(stats['columns']))}")
            
            # Print sample statistics for numeric columns if available
            if stats['sample_stats']:
                print("\n  Numeric column statistics:")
                for col, col_stats in stats['sample_stats'].items():
                    print(f"    {col}:")
                    print(f"      Mean: {col_stats['mean']:,.2f}")
                    print(f"      Min: {col_stats['min']:,.2f}")
                    print(f"      Max: {col_stats['max']:,.2f}")
                    print(f"      Std Dev: {col_stats['std']:,.2f}")
    
    # Create visualizations
    create_data_summary_visualizations(all_stats)
    
    # Save detailed statistics to CSV
    detailed_stats = pd.DataFrame()
    for category, stats in all_stats.items():
        if stats:
            category_series = pd.Series({
                'Total Records': stats['total_rows'],
                'Start Date': stats['date_range'][0] if stats['date_range'] else None,
                'End Date': stats['date_range'][1] if stats['date_range'] else None,
                'Number of Columns': len(stats['columns']),
                'Columns': ', '.join(sorted(stats['columns'])),
                'Data Types': ', '.join(f"{k}: {v}" for k, v in stats['data_types'].items())
            })
            detailed_stats[category] = category_series
    
    detailed_stats.to_csv('figures/data_analysis/detailed_data_statistics.csv')
    
    return all_stats

if __name__ == "__main__":
    print("Starting detailed data analysis...")
    all_stats = analyze_all_data()
    print("\nAnalysis complete! Results saved in figures/data_analysis/") 