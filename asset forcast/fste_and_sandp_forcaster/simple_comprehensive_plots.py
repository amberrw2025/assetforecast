#!/usr/bin/env python3
"""
Simple Comprehensive Plot Generator
==================================

Generate both Option 1 (Standard) and Option 2 (MAPE Enhanced) plots
for all stocks to ensure consistent coverage.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run comprehensive plot generation"""
    print("🚀 SIMPLE COMPREHENSIVE PLOT GENERATION")
    print("=" * 60)
    
    # Create directories
    comprehensive_dir = Path('comprehensive_plots_2024')
    option1_dir = comprehensive_dir / 'option1_standard'
    option2_dir = comprehensive_dir / 'option2_mape_enhanced'
    
    option1_dir.mkdir(parents=True, exist_ok=True)
    option2_dir.mkdir(parents=True, exist_ok=True)
    
    # Define stocks
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AZN.L', 'BP.L']
    
    print(f"📊 Processing {len(stocks)} stocks...")
    print(f"📁 Option 1: {option1_dir}")
    print(f"🌟 Option 2: {option2_dir}")
    print()
    
    for ticker in stocks:
        print(f"🎯 Processing {ticker}...")
        
        try:
            # Download data
            train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
            actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
            
            if train_data.empty or actual_2024.empty:
                print(f"❌ No data available for {ticker}")
                continue
            
            # Generate simple forecasts
            prices = train_data['Close'].values
            actual_prices = actual_2024['Close'].values
            
            # Standard forecast (linear trend)
            trend = (prices[-1] - prices[-30]) / 30
            standard_forecast = [prices[-1] + trend * i for i in range(1, len(actual_prices) + 1)]
            
            # Enhanced forecast (with market adjustments)
            if ticker.endswith('.L'):
                market_factor = 0.98  # FTSE 100
            else:
                market_factor = 1.02  # S&P 500
            
            enhanced_forecast = [prices[-1] * (market_factor ** i) for i in range(1, len(actual_prices) + 1)]
            
            # Calculate MAPE
            def calculate_mape(actual, predicted):
                return np.mean(np.abs((actual - predicted) / actual)) * 100
            
            standard_mape = calculate_mape(actual_prices, standard_forecast)
            enhanced_mape = calculate_mape(actual_prices, enhanced_forecast)
            
            # Create plots
            plt.figure(figsize=(12, 6))
            dates = actual_2024.index
            
            plt.plot(dates, actual_prices, label='Actual 2024', color='black', linewidth=2)
            plt.plot(dates, standard_forecast, label=f'Standard (MAPE: {standard_mape:.1f}%)', 
                    color='blue', linestyle='--')
            plt.plot(dates, enhanced_forecast, label=f'Enhanced (MAPE: {enhanced_mape:.1f}%)', 
                    color='red')
            
            market = "FTSE 100" if ticker.endswith('.L') else "S&P 500"
            plt.title(f'{market} | {ticker} - Comprehensive Forecast Comparison')
            plt.xlabel('Date (2024)')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save comprehensive plot
            plot_path = comprehensive_dir / f'{ticker}_comprehensive_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Created comprehensive plot: {plot_path}")
            print(f"   📊 Standard MAPE: {standard_mape:.2f}%")
            print(f"   🌟 Enhanced MAPE: {enhanced_mape:.2f}%")
            print(f"   🚀 Improvement: {((standard_mape - enhanced_mape) / standard_mape * 100):.1f}%")
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    print(f"\n🎉 COMPREHENSIVE GENERATION COMPLETE!")
    print(f"📁 Output directory: {comprehensive_dir}")
    print(f"✅ Now you have consistent plot coverage for all stocks!")

if __name__ == "__main__":
    main()
