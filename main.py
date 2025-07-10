# Author: Vu Dang Khoa

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_collection.stock_collector import StockDataCollector
from src.data_processing.data_analyzer import DataTypeAnalyzer
from src.data_processing.enhanced_predictor import EnhancedStockPredictor
from src.models.optimized_short_term_model import OptimizedShortTermModel
from src.visualization.plot_results import plot_comprehensive_results
from src.visualization.stock_dashboard import launch_dashboard
from config.settings import DEFAULT_SYMBOLS

def main():
    print("Vietnam Stock Market Analysis & Prediction System")
    print("=" * 60)
    
    while True:
        print("\nSelect an option:")
        print("1. Collect Stock Data")
        print("2. Analyze Data Types")
        print("3. Train Prediction Model")
        print("4. Generate Results & Plots")
        print("5. Launch Dashboard")
        print("6. Run Complete Pipeline")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            collector = StockDataCollector()
            collector.collect_all()
            print("Data collection completed!")
            
        elif choice == '2':
            analyzer = DataTypeAnalyzer()
            for symbol in DEFAULT_SYMBOLS:
                print(f"\n📊 Analyzing {symbol}...")
                analysis = analyzer.get_report(symbol)
                analyzer.print_analysis(analysis)
            print("Data analysis completed!")
            
        elif choice == '3':
            predictor = EnhancedStockPredictor()
            for symbol in DEFAULT_SYMBOLS:
                print(f"🧠 Training model for {symbol}...")
                predictor.train_model(symbol)
            print("Model training completed!")
            
        elif choice == '4':
            print("📈 Generating plots and results...")
            plot_comprehensive_results()
            print("Plots generated!")
            
        elif choice == '5':
            print("Launching dashboard...")
            launch_dashboard()
            
        elif choice == '6':
            print("Running complete pipeline...")
            
            collector = StockDataCollector()
            collector.collect_all()
            print("Step 1/4: Data collection completed!")
            
            analyzer = DataTypeAnalyzer()
            for symbol in DEFAULT_SYMBOLS:
                analysis = analyzer.get_report(symbol)
            print("Step 2/4: Data analysis completed!")
            
            predictor = EnhancedStockPredictor()
            for symbol in DEFAULT_SYMBOLS:
                predictor.train_model(symbol)
            print("Step 3/4: Model training completed!")
            
            plot_comprehensive_results()
            print("Step 4/4: Results generated!")
            
            print("Complete pipeline finished successfully!")
            
        elif choice == '0':
            print("Goodbye!")
            print("Author: Vu Dang Khoa")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
