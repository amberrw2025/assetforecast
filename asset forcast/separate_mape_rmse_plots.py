#!/usr/bin/env python3
"""
Separate MAPE and RMSE Standard Forecast Plots
=============================================

Creates TWO separate plots for each stock:
1. MAPE-focused plot: Shows percentage-based forecast accuracy
2. RMSE-focused plot: Shows absolute error-based forecast accuracy

This helps visualize how different metrics evaluate the same forecasts.
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
    """Set the unified visual style for all plots"""
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

def create_mape_focused_plot(ticker, train_data, actual_2024, old_forecast, enhanced_forecast, forecast_dates):
    """Create a plot focused on MAPE evaluation"""
    
    # Calculate MAPE metrics
    mape_old = mape_new = mape_improvement_pct = 0
    
    if not actual_2024.empty:
        # Handle new yfinance format
        if isinstance(actual_2024['Close'], pd.DataFrame):
            actual_prices = actual_2024['Close'].iloc[:, 0].values
        else:
            actual_prices = actual_2024['Close'].values
        
        overlap_days = min(len(actual_prices), len(old_forecast))
        
        if overlap_days > 0:
            actual_subset = actual_prices[:overlap_days]
            old_subset = old_forecast[:overlap_days]
            enhanced_subset = enhanced_forecast[:overlap_days]
            
            # Calculate MAPE
            mape_old = np.mean(np.abs((actual_subset - old_subset) / actual_subset)) * 100
            mape_new = np.mean(np.abs((actual_subset - enhanced_subset) / actual_subset)) * 100
            
            mape_improvement = mape_old - mape_new
            mape_improvement_pct = (mape_improvement / mape_old) * 100 if mape_old != 0 else 0
    
    # MAPE-FOCUSED PLOT
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Historical data (last 3 months)
    recent_data = train_data.tail(63)
    if isinstance(recent_data['Close'], pd.DataFrame):
        close_prices = recent_data['Close'].iloc[:, 0]
    else:
        close_prices = recent_data['Close']
    
    ax.plot(recent_data.index, close_prices, 
            label='Historical Data', 
            color='#666666', alpha=0.8, linewidth=1.5)
    
    # OLD Method
    ax.plot(forecast_dates, old_forecast, 
            label=f'OLD Method (MAPE: {mape_old:.1f}%)', 
            color='#1f77b4', linestyle='--', linewidth=2)
    
    # NEW Enhanced
    ax.plot(forecast_dates, enhanced_forecast, 
            label=f'NEW Enhanced (MAPE: {mape_new:.1f}%)', 
            color='#ff7f0e', linestyle='-.', linewidth=2)
    
    # Actual 2024 Prices
    if not actual_2024.empty:
        if isinstance(actual_2024['Close'], pd.DataFrame):
            actual_close_prices = actual_2024['Close'].iloc[:, 0]
        else:
            actual_close_prices = actual_2024['Close']
        
        ax.plot(actual_2024.index, actual_close_prices, 
                label='Actual 2024 Prices', 
                color='#000000', linewidth=2.5)
    
    # Formatting
    market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
    
    # MAPE-focused title
    if mape_improvement_pct != 0:
        improvement_text = f" | MAPE Improvement: {mape_improvement_pct:+.1f}%"
    else:
        improvement_text = ""
    
    ax.set_title(f'{ticker} - MAPE Evaluation{improvement_text}', 
                fontsize=12, fontweight='normal', pad=10)
    fig.suptitle(f'{market} | {ticker} - Percentage Error Analysis', 
                fontsize=14, fontweight='bold', y=0.98)
    
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Spines styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save MAPE plot
    output_dir = Path('separate_metric_plots/mape_focused')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{ticker}_MAPE_evaluation.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    
    return mape_old, mape_new

def create_rmse_focused_plot(ticker, train_data, actual_2024, old_forecast, enhanced_forecast, forecast_dates):
    """Create a plot focused on RMSE evaluation"""
    
    # Calculate RMSE metrics
    rmse_old = rmse_new = rmse_improvement_pct = 0
    
    if not actual_2024.empty:
        # Handle new yfinance format
        if isinstance(actual_2024['Close'], pd.DataFrame):
            actual_prices = actual_2024['Close'].iloc[:, 0].values
        else:
            actual_prices = actual_2024['Close'].values
        
        overlap_days = min(len(actual_prices), len(old_forecast))
        
        if overlap_days > 0:
            actual_subset = actual_prices[:overlap_days]
            old_subset = old_forecast[:overlap_days]
            enhanced_subset = enhanced_forecast[:overlap_days]
            
            # Calculate RMSE
            rmse_old = np.sqrt(np.mean((actual_subset - old_subset) ** 2))
            rmse_new = np.sqrt(np.mean((actual_subset - enhanced_subset) ** 2))
            
            rmse_improvement = rmse_old - rmse_new
            rmse_improvement_pct = (rmse_improvement / rmse_old) * 100 if rmse_old != 0 else 0
    
    # RMSE-FOCUSED PLOT
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Historical data (last 3 months)
    recent_data = train_data.tail(63)
    if isinstance(recent_data['Close'], pd.DataFrame):
        close_prices = recent_data['Close'].iloc[:, 0]
    else:
        close_prices = recent_data['Close']
    
    ax.plot(recent_data.index, close_prices, 
            label='Historical Data', 
            color='#666666', alpha=0.8, linewidth=1.5)
    
    # OLD Method
    ax.plot(forecast_dates, old_forecast, 
            label=f'OLD Method (RMSE: {rmse_old:.2f})', 
            color='#1f77b4', linestyle='--', linewidth=2)
    
    # NEW Enhanced
    ax.plot(forecast_dates, enhanced_forecast, 
            label=f'NEW Enhanced (RMSE: {rmse_new:.2f})', 
            color='#ff7f0e', linestyle='-.', linewidth=2)
    
    # Actual 2024 Prices
    if not actual_2024.empty:
        if isinstance(actual_2024['Close'], pd.DataFrame):
            actual_close_prices = actual_2024['Close'].iloc[:, 0]
        else:
            actual_close_prices = actual_2024['Close']
        
        ax.plot(actual_2024.index, actual_close_prices, 
                label='Actual 2024 Prices', 
                color='#000000', linewidth=2.5)
    
    # Formatting
    market = "🇬🇧 FTSE 100" if ticker.endswith('.L') else "🇺🇸 S&P 500"
    
    # RMSE-focused title
    if rmse_improvement_pct != 0:
        improvement_text = f" | RMSE Improvement: {rmse_improvement_pct:+.1f}%"
    else:
        improvement_text = ""
    
    ax.set_title(f'{ticker} - RMSE Evaluation{improvement_text}', 
                fontsize=12, fontweight='normal', pad=10)
    fig.suptitle(f'{market} | {ticker} - Absolute Error Analysis', 
                fontsize=14, fontweight='bold', y=0.98)
    
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Spines styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save RMSE plot
    output_dir = Path('separate_metric_plots/rmse_focused')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{ticker}_RMSE_evaluation.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    
    return rmse_old, rmse_new

def create_dual_metric_plots(ticker):
    """Create both MAPE and RMSE focused plots for a ticker"""
    
    print(f"📊 Creating dual plots for {ticker}...")
    
    try:
        # Download data
        train_data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
        actual_2024 = yf.download(ticker, start='2024-01-01', end='2024-06-30', progress=False)
        
        if train_data.empty:
            print(f"   ❌ No training data available for {ticker}")
            return False
        
        # Handle new yfinance format
        if isinstance(train_data['Close'], pd.DataFrame):
            prices = train_data['Close'].iloc[:, 0].values
        else:
            prices = train_data['Close'].values
        
        # Calculate trend (same as standard forecast)
        if len(prices) >= 30:
            trend = (prices[-1] - prices[-30]) / 30
        else:
            trend = (prices[-1] - prices[0]) / len(prices)
        
        # Generate forecasts
        forecast_days = 126
        
        # OLD Method (simple linear trend)
        old_forecast = []
        for i in range(1, forecast_days + 1):
            predicted_price = prices[-1] + (trend * i)
            old_forecast.append(max(0.01, predicted_price))
        
        # NEW Enhanced (with slight enhancement)
        enhanced_forecast = []
        for i in range(1, forecast_days + 1):
            volatility_factor = 1 + (0.001 * i)
            predicted_price = prices[-1] + (trend * i * volatility_factor)
            enhanced_forecast.append(max(0.01, predicted_price))
        
        # Create forecast dates
        last_date = train_data.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Create MAPE-focused plot
        mape_old, mape_new = create_mape_focused_plot(
            ticker, train_data, actual_2024, old_forecast, enhanced_forecast, forecast_dates)
        
        # Create RMSE-focused plot
        rmse_old, rmse_new = create_rmse_focused_plot(
            ticker, train_data, actual_2024, old_forecast, enhanced_forecast, forecast_dates)
        
        print(f"   ✅ Dual plots created for {ticker}")
        print(f"      MAPE: OLD={mape_old:.1f}%, NEW={mape_new:.1f}%")
        print(f"      RMSE: OLD={rmse_old:.2f}, NEW={rmse_new:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating plots for {ticker}: {e}")
        return False

def main():
    """Main execution function"""
    print("📊 SEPARATE MAPE & RMSE STANDARD FORECAST PLOTS")
    print("=" * 60)
    print("🎯 Creating TWO plots per stock:")
    print("   📈 MAPE-focused plot (percentage error analysis)")
    print("   📉 RMSE-focused plot (absolute error analysis)")
    print("🔍 This shows how different metrics evaluate the same forecasts")
    print()
    
    # Set unified plot style
    set_unified_plot_style()
    
    # Sample tickers for demonstration
    sample_tickers = [
        # US Stocks (mix of price levels)
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        'JPM', 'JNJ', 'PG', 'KO', 'DIS',
        # UK Stocks (FTSE 100)
        'BARC.L', 'BP.L', 'LLOY.L', 'VOD.L', 'TSCO.L',
        'AZN.L', 'RIO.L', 'SHEL.L', 'ULVR.L', 'BT-A.L'
    ]
    
    print(f"🚀 Processing {len(sample_tickers)} tickers...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    failed_tickers = []
    
    for i, ticker in enumerate(sample_tickers, 1):
        print(f"\n[{i:2d}/{len(sample_tickers)}]", end=" ")
        if create_dual_metric_plots(ticker):
            successful += 1
        else:
            failed += 1
            failed_tickers.append(ticker)
    
    print("\n" + "=" * 60)
    print("📊 SEPARATE METRIC PLOTS GENERATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully processed: {successful} tickers")
    print(f"❌ Failed: {failed} tickers")
    
    if failed_tickers:
        print(f"\n⚠️  Failed tickers: {', '.join(failed_tickers)}")
    
    print(f"\n🎉 GENERATED PLOTS:")
    print(f"   📁 MAPE plots: separate_metric_plots/mape_focused/")
    print(f"   📁 RMSE plots: separate_metric_plots/rmse_focused/")
    
    print(f"\n💡 KEY DIFFERENCES:")
    print(f"   📈 MAPE plots: Show percentage-based accuracy (scale-independent)")
    print(f"   📉 RMSE plots: Show absolute error in price units (scale-dependent)")
    print(f"   🔍 Use MAPE for cross-stock comparison")
    print(f"   🔍 Use RMSE for understanding actual price deviation")
    
    print(f"\n✨ Each stock now has TWO evaluation perspectives!")

if __name__ == "__main__":
    main() 