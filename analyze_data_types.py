import pandas as pd
import os
from datetime import datetime
import numpy as np

def analyze_file_structure(file_path):
    """Analyze the structure and data types of a CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get basic information
        print(f"\nFile: {os.path.basename(file_path)}")
        print("=" * 50)
        print("\nColumns and Data Types:")
        
        # First, show DataFrame info
        print("\nDataFrame Info:")
        df.info(show_counts=True)
        
        print("\nDetailed Column Analysis:")
        for col in df.columns:
            # Get unique values sample
            unique_sample = df[col].dropna().unique()[:3]
            
            print(f"\n{col}:")
            print(f"- Python dtype: {df[col].dtype}")
            print(f"- Sample values: {unique_sample}")
            
            # Additional type checking for specific columns
            if 'time' in col.lower():
                try:
                    # Try parsing as datetime
                    if '+' in str(df[col].iloc[0]):  # Check if timezone info exists
                        pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S%z')
                        print("- Confirmed datetime with timezone")
                    else:
                        pd.to_datetime(df[col])
                        print("- Confirmed datetime")
                except Exception as e:
                    print(f"- Time format could not be parsed: {str(e)}")
            
            # Check numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dtype == np.int64:
                    print("- Integer values")
                else:
                    # Check if all float values are actually integers
                    non_na = df[col].dropna()
                    if all(float(x).is_integer() for x in non_na):
                        print("- Float values (all integers)")
                    else:
                        print("- Float values with decimals")
                print(f"- Range: [{df[col].min()}, {df[col].max()}]")
        
        # Show some statistical information for numeric columns
        print("\nNumerical Columns Statistics:")
        print(df.describe())
        
        print(f"\nTotal rows: {len(df)}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")

def main():
    base_dir = "data/collected_data"
    
    # Analyze daily data
    print("\nANALYZING DAILY DATA")
    print("*" * 80)
    daily_dir = os.path.join(base_dir, "daily")
    for file in os.listdir(daily_dir):
        if file.endswith('.csv'):
            analyze_file_structure(os.path.join(daily_dir, file))
    
    # Analyze intraday data
    print("\nANALYZING INTRADAY DATA")
    print("*" * 80)
    intraday_dir = os.path.join(base_dir, "intraday")
    for file in os.listdir(intraday_dir):
        if file.endswith('.csv'):
            analyze_file_structure(os.path.join(intraday_dir, file))

if __name__ == "__main__":
    main() 