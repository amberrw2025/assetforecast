#!/usr/bin/env python3
"""
Red/Green/Black Standardization Script
=====================================

Complete visual standardization for ALL standard forecast plots.
This script ensures every single standard forecast graph has identical:
- Color scheme: Green historical, Red forecast, Black actual
- Visual formatting: Grid, fonts, legends, layout
- Forecasting methodology: Consistent algorithms across all stocks

Eliminates the visual inconsistency problem once and for all.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def set_unified_plot_style():
    """Set the unified visual style for all standard forecast plots"""
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
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True
    })

def create_standardized_plot(ticker):
    """Create a plot with unified red/green/black visual style"""
    
    print(f"🎯 Standardizing {ticker} with red/green/black scheme...")
    
    try:
        # Download data with same approach as working AAPL script
        train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
        actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
        
        if train_data.empty:
            print(f"   ❌ No data available for {ticker}")
            return False
        
        # Use exact forecasting methodology from working AAPL script  
        # Handle new yfinance format that returns DataFrame for single column
        if isinstance(train_data['Close'], pd.DataFrame):
            prices = train_data['Close'].iloc[:, 0].values
        else:
            prices = train_data['Close'].values
        
        # AAPL-style trend calculation (30-day trend)
        if len(prices) >= 30:
            trend = (prices[-1] - prices[-30]) / 30
        else:
            trend = (prices[-1] - prices[0]) / len(prices)
        
        # Generate forecast exactly like working AAPL script
        forecast_days = 126
        forecast = []
        for i in range(1, forecast_days + 1):
            # Simple linear progression (AAPL style)
            predicted_price = prices[-1] + (trend * i)
            forecast.append(max(0.01, predicted_price))
        
        # Create forecast dates (business days only, like AAPL)
        last_date = train_data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # RED/GREEN/BLACK PLOT CREATION (using AAPL structure)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Historical data (GREEN instead of gray)
        recent_data = train_data.tail(63)
        # Handle new yfinance format
        if isinstance(recent_data['Close'], pd.DataFrame):
            close_prices = recent_data['Close'].iloc[:, 0]
        else:
            close_prices = recent_data['Close']
        ax.plot(recent_data.index, close_prices, 
                label='Historical', 
                color='#2E8B57',  # Sea Green instead of gray
                alpha=0.8, 
                linewidth=1.5)
        
        # 2. Forecast (RED instead of blue)
        ax.plot(forecast_dates, forecast, 
                label='2024 Standard Forecast', 
                color='#DC143C',  # Crimson Red instead of blue
                linestyle='--',   # Keep dashed style
                linewidth=2)
        
        # 3. Actual 2024 data (BLACK - same as AAPL)
        mape_text = ""
        if not actual_2024.empty:
            # Handle new yfinance format
            if isinstance(actual_2024['Close'], pd.DataFrame):
                actual_close_prices = actual_2024['Close'].iloc[:, 0]
                actual_prices = actual_2024['Close'].iloc[:, 0].values
            else:
                actual_close_prices = actual_2024['Close']
                actual_prices = actual_2024['Close'].values
            
            ax.plot(actual_2024.index, actual_close_prices, 
                    label='Actual 2024', 
                    color='#000000',  # Keep black
                    linewidth=2.5)
            
            # Calculate MAPE (same as AAPL style)
            overlap_days = min(len(actual_prices), len(forecast))
            
            if overlap_days > 0:
                forecast_subset = forecast[:overlap_days]
                actual_subset = actual_prices[:overlap_days]
                mape = np.mean(np.abs((actual_subset - forecast_subset) / actual_subset)) * 100
                mape_text = f" (MAPE: {mape:.1f}%)"
        
        # 4. EXACT AAPL-STYLE FORMATTING (just with new colors)
        market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
        
        # Title exactly like AAPL
        title = f'{ticker} - Standard Forecast vs Actual{mape_text}'
        ax.set_title(title, fontsize=12, fontweight='normal', pad=10)
        fig.suptitle(f'{market} | {ticker}', fontsize=14, fontweight='bold', y=0.98)
        
        # Axes labels exactly like AAPL
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        
        # Legend exactly like AAPL
        ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Grid exactly like AAPL
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Spines exactly like AAPL
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Layout exactly like AAPL
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Space for suptitle
        
        # Save with AAPL-style settings
        output_dir = Path('comprehensive_plots_2024/option1_standard')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{ticker}_2024_vs_actual.png'
        
        plt.savefig(output_path, 
                   dpi=300, 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.1)
        plt.close()
        
        print(f"   ✅ Standardized plot created: {ticker}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error standardizing {ticker}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 RED/GREEN/BLACK STANDARDIZATION")
    print("=" * 60)
    print("🎯 Goal: Complete visual consistency for ALL standard forecast plots")
    print("🎨 Color Scheme: Green historical | Red forecast | Black actual")
    print("📊 Methodology: Unified forecasting algorithm across all stocks")
    print()
    
    # Set unified plot style
    set_unified_plot_style()
    
    # Get all existing tickers from option1 directory
    option1_dir = Path('comprehensive_plots_2024/option1_standard')
    
    all_tickers = set()
    if option1_dir.exists():
        for file in option1_dir.glob('*.png'):
            ticker = file.stem.replace('_2024_vs_actual', '').replace('_REAL_', '')
            all_tickers.add(ticker)
    
    all_tickers = sorted(list(all_tickers))
    
    print(f"📈 Found {len(all_tickers)} tickers to standardize:")
    print("-" * 40)
    for i, ticker in enumerate(all_tickers, 1):
        market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
        print(f"   {i:2d}. {ticker:12s} ({market})")
    
    print(f"\n🚀 Starting red/green/black standardization...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    failed_tickers = []
    
    for i, ticker in enumerate(all_tickers, 1):
        print(f"\n[{i:2d}/{len(all_tickers)}]", end=" ")
        if create_standardized_plot(ticker):
            successful += 1
        else:
            failed += 1
            failed_tickers.append(ticker)
    
    print("\n" + "=" * 60)
    print("📊 STANDARDIZATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully standardized: {successful} plots")
    print(f"❌ Failed: {failed} plots")
    
    if failed_tickers:
        print(f"\n⚠️  Failed tickers: {', '.join(failed_tickers)}")
    
    print(f"\n🎉 ALL STANDARD FORECAST PLOTS NOW HAVE:")
    print(f"   🟢 Green lines for historical data")
    print(f"   🔴 Red dashed lines for forecasts") 
    print(f"   ⚫ Black solid lines for actual 2024 data")
    print(f"   📐 Identical formatting and layout")
    print(f"   🔬 Consistent forecasting methodology")
    
    print(f"\n💡 Visual inconsistency problem SOLVED!")
    print("   All standard forecast graphs now look identical.")

if __name__ == "__main__":
    main() 