#!/usr/bin/env python3
"""
AAPL Style Standardization
=========================

Make ALL standard forecast plots look exactly like AAPL's style.
This ensures consistent visual appearance across all stocks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def set_aapl_plot_style():
    """Set the exact plot style that AAPL uses"""
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

def create_aapl_style_plot(ticker):
    """Create a plot with AAPL's exact visual style"""
    
    print(f"🎯 Creating AAPL-style plot for {ticker}...")
    
    try:
        # Download data with same approach as AAPL
        train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
        actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
        
        if train_data.empty:
            print(f"   ❌ No data available for {ticker}")
            return False
        
        # Use AAPL's exact forecasting methodology
        prices = train_data['Close'].values
        
        # AAPL-style trend calculation (30-day trend)
        if len(prices) >= 30:
            trend = (prices[-1] - prices[-30]) / 30
        else:
            trend = (prices[-1] - prices[0]) / len(prices)
        
        # Generate forecast exactly like AAPL
        forecast_days = 126
        forecast = []
        for i in range(1, forecast_days + 1):
            # Simple linear progression (AAPL style)
            predicted_price = prices[-1] + (trend * i)
            forecast.append(max(0.01, predicted_price))
        
        # Create forecast dates (business days only, like AAPL)
        last_date = train_data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # AAPL-STYLE PLOT CREATION
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Historical data (AAPL style - last 3 months)
        recent_data = train_data.tail(63)
        ax.plot(recent_data.index, recent_data['Close'], 
                label='Historical', 
                color='#666666',  # AAPL's exact gray color
                alpha=0.8, 
                linewidth=1.5)
        
        # 2. Forecast (AAPL style - blue dashed line)
        ax.plot(forecast_dates, forecast, 
                label='2024 Standard Forecast', 
                color='#1f77b4',  # AAPL's exact blue
                linestyle='--',   # AAPL's dashed style
                linewidth=2)
        
        # 3. Actual 2024 data (AAPL style - black solid line)
        mape_text = ""
        if not actual_2024.empty:
            ax.plot(actual_2024.index, actual_2024['Close'], 
                    label='Actual 2024', 
                    color='#000000',  # AAPL's exact black
                    linewidth=2.5)
            
            # Calculate MAPE (AAPL style)
            actual_prices = actual_2024['Close'].values
            overlap_days = min(len(actual_prices), len(forecast))
            
            if overlap_days > 0:
                forecast_subset = forecast[:overlap_days]
                actual_subset = actual_prices[:overlap_days]
                mape = np.mean(np.abs((actual_subset - forecast_subset) / actual_subset)) * 100
                mape_text = f" (MAPE: {mape:.1f}%)"
        
        # 4. AAPL-STYLE FORMATTING
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
        
        print(f"   ✅ AAPL-style plot created: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating AAPL-style plot for {ticker}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 AAPL STYLE STANDARDIZATION")
    print("=" * 50)
    print("🎯 Goal: Make ALL standard plots look exactly like AAPL")
    print()
    
    # Set AAPL plot style
    set_aapl_plot_style()
    
    # Get all existing tickers from option1 directory
    option1_dir = Path('comprehensive_plots_2024/option1_standard')
    
    all_tickers = set()
    if option1_dir.exists():
        for file in option1_dir.glob('*.png'):
            ticker = file.stem.replace('_2024_vs_actual', '').replace('_REAL_', '')
            all_tickers.add(ticker)
    
    all_tickers = sorted(list(all_tickers))
    
    print(f"📊 Found {len(all_tickers)} tickers to standardize to AAPL style:")
    for i, ticker in enumerate(all_tickers, 1):
        market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
        print(f"   {i:2d}. {ticker:10s} ({market})")
    
    print(f"\n🚀 Starting AAPL-style standardization...")
    print("-" * 50)
    
    successful = 0
    failed = 0
    
    for ticker in all_tickers:
        if create_aapl_style_plot(ticker):
            successful += 1
        else:
            failed += 1
        print()  # Add spacing
    
    print("=" * 50)
    print("🎉 AAPL STYLE STANDARDIZATION COMPLETE!")
    print(f"✅ Successfully styled: {successful} stocks")
    print(f"❌ Failed: {failed} stocks")
    print()
    print("🚀 RESULT: All standard forecast plots now look exactly like AAPL!")
    print("   • Same colors and line styles")
    print("   • Same layout and formatting")
    print("   • Same legend and grid appearance")
    print("   • Consistent visual hierarchy")

if __name__ == "__main__":
    main() 