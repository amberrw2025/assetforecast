#!/usr/bin/env python3
"""
Create Comprehensive Plot Summary
=================================

Organizes existing plots and creates a comprehensive summary showing both plot types.
"""

import os
import shutil
from pathlib import Path
import pandas as pd

def create_comprehensive_summary():
    """Create comprehensive summary of both plot types"""
    
    print("🎯 CREATING COMPREHENSIVE PLOT SUMMARY")
    print("=" * 50)
    
    # Create comprehensive directory structure
    comprehensive_dir = Path('comprehensive_plots_2024')
    option1_dir = comprehensive_dir / 'option1_standard'
    option2_dir = comprehensive_dir / 'option2_mape_enhanced'
    
    comprehensive_dir.mkdir(exist_ok=True)
    option1_dir.mkdir(exist_ok=True)
    option2_dir.mkdir(exist_ok=True)
    
    # Find existing plots
    print("📊 Finding existing plots...")
    
    # Option 1 plots (standard)
    option1_plots = list(Path('.').glob('*_2024_vs_actual.png'))
    print(f"   📊 Found {len(option1_plots)} Option 1 plots (standard)")
    
    # Option 2 plots (MAPE enhanced)
    option2_plots = list(Path('unified_mape_plots_2024').glob('*_unified_mape_forecast_2024.png'))
    print(f"   🌟 Found {len(option2_plots)} Option 2 plots (MAPE enhanced)")
    
    # Copy Option 1 plots
    print("\n📁 Organizing Option 1 plots (standard)...")
    for plot in option1_plots:
        ticker = plot.stem.replace('_2024_vs_actual', '')
        new_path = option1_dir / f'{ticker}_2024_vs_actual.png'
        shutil.copy2(plot, new_path)
        print(f"   ✅ Copied {ticker}")
    
    # Copy Option 2 plots
    print("\n📁 Organizing Option 2 plots (MAPE enhanced)...")
    for plot in option2_plots:
        ticker = plot.stem.replace('_unified_mape_forecast_2024', '')
        new_path = option2_dir / f'{ticker}_unified_mape_forecast_2024.png'
        shutil.copy2(plot, new_path)
        print(f"   ✅ Copied {ticker}")
    
    # Create summary report
    print("\n📋 Creating summary report...")
    
    summary_data = []
    
    # Get all unique tickers
    all_tickers = set()
    for plot in option1_plots:
        ticker = plot.stem.replace('_2024_vs_actual', '')
        all_tickers.add(ticker)
    
    for plot in option2_plots:
        ticker = plot.stem.replace('_unified_mape_forecast_2024', '')
        all_tickers.add(ticker)
    
    # Create summary for each ticker
    for ticker in sorted(all_tickers):
        option1_exists = (option1_dir / f'{ticker}_2024_vs_actual.png').exists()
        option2_exists = (option2_dir / f'{ticker}_unified_mape_forecast_2024.png').exists()
        
        market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
        
        summary_data.append({
            'Ticker': ticker,
            'Market': market,
            'Option 1 (Standard)': '✅ Available' if option1_exists else '❌ Missing',
            'Option 2 (MAPE Enhanced)': '✅ Available' if option2_exists else '❌ Missing',
            'Both Available': '✅ Yes' if option1_exists and option2_exists else '❌ No'
        })
    
    # Create summary DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = comprehensive_dir / 'plot_summary.csv'
    df.to_csv(summary_path, index=False)
    
    # Print summary
    print(f"\n📊 PLOT SUMMARY:")
    print("=" * 50)
    print(df.to_string(index=False))
    
    # Statistics
    total_stocks = len(all_tickers)
    option1_count = sum(1 for row in summary_data if 'Available' in row['Option 1 (Standard)'])
    option2_count = sum(1 for row in summary_data if 'Available' in row['Option 2 (MAPE Enhanced)'])
    both_count = sum(1 for row in summary_data if 'Yes' in row['Both Available'])
    
    print(f"\n📈 STATISTICS:")
    print(f"   📊 Total stocks: {total_stocks}")
    print(f"   📊 Option 1 (Standard): {option1_count}/{total_stocks}")
    print(f"   🌟 Option 2 (MAPE Enhanced): {option2_count}/{total_stocks}")
    print(f"   ✅ Both available: {both_count}/{total_stocks}")
    
    # Market breakdown
    ftse_stocks = [row for row in summary_data if 'FTSE' in row['Market']]
    sp500_stocks = [row for row in summary_data if 'S&P' in row['Market']]
    
    ftse_option1 = sum(1 for row in ftse_stocks if 'Available' in row['Option 1 (Standard)'])
    ftse_option2 = sum(1 for row in ftse_stocks if 'Available' in row['Option 2 (MAPE Enhanced)'])
    sp500_option1 = sum(1 for row in sp500_stocks if 'Available' in row['Option 1 (Standard)'])
    sp500_option2 = sum(1 for row in sp500_stocks if 'Available' in row['Option 2 (MAPE Enhanced)'])
    
    print(f"\n🇬🇧 FTSE 100:")
    print(f"   📊 Option 1: {ftse_option1}/{len(ftse_stocks)}")
    print(f"   🌟 Option 2: {ftse_option2}/{len(ftse_stocks)}")
    
    print(f"\n🇺🇸 S&P 500:")
    print(f"   📊 Option 1: {sp500_option1}/{len(sp500_stocks)}")
    print(f"   🌟 Option 2: {sp500_option2}/{len(sp500_stocks)}")
    
    print(f"\n💾 Files saved:")
    print(f"   📁 Comprehensive directory: {comprehensive_dir}")
    print(f"   📊 Option 1 plots: {option1_dir}")
    print(f"   🌟 Option 2 plots: {option2_dir}")
    print(f"   📋 Summary report: {summary_path}")
    
    print(f"\n✅ COMPREHENSIVE SUMMARY COMPLETE!")
    print("You now have both plot types organized and summarized!")

if __name__ == "__main__":
    create_comprehensive_summary() 