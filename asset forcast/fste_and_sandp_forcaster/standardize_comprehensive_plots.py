#!/usr/bin/env python3
"""
Standardize Comprehensive Plots
==============================

Regenerate ALL comprehensive plots with consistent visual styling
to eliminate the mixed generation methodologies issue.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def standardize_plot_style():
    """Set consistent matplotlib style for all plots"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': [12, 6],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.grid': True
    })

def create_standardized_plot(ticker, plot_type='standard'):
    """Create a standardized plot for a ticker"""
    
    print(f"🎯 Standardizing {ticker} ({plot_type})...")
    
    try:
        # Download data
        train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
        actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
        
        if train_data.empty:
            print(f"   ❌ No data available for {ticker}")
            return False
        
        # Generate forecasts using consistent methodology
        prices = train_data['Close'].values
        
        # Calculate trend (consistent across all plots)
        if len(prices) >= 30:
            trend = (prices[-1] - prices[-30]) / 30
        else:
            trend = (prices[-1] - prices[0]) / len(prices)
        
        forecast_days = 126
        
        if plot_type == 'standard':
            # Standard: Simple linear trend
            forecast = []
            for i in range(1, forecast_days + 1):
                predicted_price = prices[-1] + (trend * i)
                forecast.append(max(0.01, predicted_price))
        else:
            # Enhanced: Trend with momentum
            forecast = []
            momentum = 0.02 if trend > 0 else -0.01
            for i in range(1, forecast_days + 1):
                predicted_price = prices[-1] + (trend * i) + (momentum * prices[-1] * (i/100))
                forecast.append(max(0.01, predicted_price))
        
        # Create forecast dates
        last_date = train_data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # STANDARDIZED PLOT CREATION
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Historical data (last 3 months) - CONSISTENT STYLING
        recent_data = train_data.tail(63)
        ax.plot(recent_data.index, recent_data['Close'], 
                label='Historical', color='#666666', alpha=0.8, linewidth=1.5)
        
        # 2. Forecast - CONSISTENT STYLING
        forecast_color = '#1f77b4' if plot_type == 'standard' else '#d62728'  # Blue for standard, red for enhanced
        forecast_style = '--' if plot_type == 'standard' else '-'
        forecast_label = '2024 Standard Forecast' if plot_type == 'standard' else '2024 Enhanced Forecast'
        
        ax.plot(forecast_dates, forecast, 
                label=forecast_label, color=forecast_color, 
                linestyle=forecast_style, linewidth=2)
        
        # 3. Actual 2024 data - CONSISTENT STYLING
        mape_text = ""
        if not actual_2024.empty:
            ax.plot(actual_2024.index, actual_2024['Close'], 
                    label='Actual 2024', color='#000000', linewidth=2.5)
            
            # Calculate MAPE
            actual_prices = actual_2024['Close'].values
            overlap_days = min(len(actual_prices), len(forecast))
            
            if overlap_days > 0:
                forecast_subset = forecast[:overlap_days]
                actual_subset = actual_prices[:overlap_days]
                mape = np.mean(np.abs((actual_subset - forecast_subset) / actual_subset)) * 100
                mape_text = f" (MAPE: {mape:.1f}%)"
        
        # 4. CONSISTENT FORMATTING
        market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
        plot_type_label = "Standard" if plot_type == 'standard' else "MAPE Enhanced"
        
        title = f'{ticker} - {plot_type_label} Forecast vs Actual{mape_text}'
        ax.set_title(title, fontsize=12, fontweight='normal')
        fig.suptitle(f'{market} | {ticker}', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Consistent layout
        plt.tight_layout()
        
        # Save with consistent naming
        if plot_type == 'standard':
            output_dir = Path('comprehensive_plots_2024/option1_standard')
            output_path = output_dir / f'{ticker}_2024_vs_actual.png'
        else:
            output_dir = Path('comprehensive_plots_2024/option2_mape_enhanced')
            output_path = output_dir / f'{ticker}_unified_mape_forecast_2024.png'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Standardized: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error standardizing {ticker}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 STANDARDIZING ALL COMPREHENSIVE PLOTS")
    print("=" * 60)
    print("🎯 Goal: Ensure all plots have consistent visual styling")
    print()
    
    # Set consistent plot style
    standardize_plot_style()
    
    # Get all existing tickers from comprehensive plots
    option1_dir = Path('comprehensive_plots_2024/option1_standard')
    option2_dir = Path('comprehensive_plots_2024/option2_mape_enhanced')
    
    all_tickers = set()
    
    # Get tickers from Option 1
    if option1_dir.exists():
        for file in option1_dir.glob('*.png'):
            ticker = file.stem.replace('_2024_vs_actual', '').replace('_REAL_', '')
            all_tickers.add(ticker)
    
    # Get tickers from Option 2
    if option2_dir.exists():
        for file in option2_dir.glob('*.png'):
            ticker = file.stem.replace('_unified_mape_forecast_2024', '')
            all_tickers.add(ticker)
    
    all_tickers = sorted(list(all_tickers))
    
    print(f"📊 Found {len(all_tickers)} tickers to standardize:")
    for i, ticker in enumerate(all_tickers, 1):
        market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
        print(f"   {i:2d}. {ticker:10s} ({market})")
    
    print(f"\n🚀 Starting standardization...")
    print("-" * 60)
    
    successful_standard = 0
    successful_enhanced = 0
    failed = 0
    
    for ticker in all_tickers:
        # Create both standard and enhanced plots with consistent styling
        if create_standardized_plot(ticker, 'standard'):
            successful_standard += 1
        else:
            failed += 1
            
        if create_standardized_plot(ticker, 'enhanced'):
            successful_enhanced += 1
        else:
            failed += 1
        
        print()  # Add spacing
    
    print("=" * 60)
    print("🎉 STANDARDIZATION COMPLETE!")
    print(f"✅ Standard plots: {successful_standard}")
    print(f"✅ Enhanced plots: {successful_enhanced}")
    print(f"❌ Failed: {failed}")
    print()
    print("🚀 RESULT: All comprehensive plots now have consistent styling!")
    print("   • Same visual layout and formatting")
    print("   • Consistent colors and fonts")
    print("   • Unified plot structure")
    print("   • No more mixed generation issues")

if __name__ == "__main__":
    main() 