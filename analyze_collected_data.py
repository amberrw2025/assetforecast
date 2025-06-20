#!/usr/bin/env python3
"""
Data Analysis Script - NO MODEL TRAINING
Analyzes the collected 10-year historical stock data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_collected_data():
    """Analyze the collected historical data without training any models"""
    
    print("🔍 ANALYZING 10-YEAR HISTORICAL STOCK DATA")
    print("⚠️  NO MODEL TRAINING - DATA ANALYSIS ONLY")
    print("="*60)
    
    # Load the combined dataset
    data_file = Path("data/historical_10years/all_stocks_10year_combined.csv")
    
    if not data_file.exists():
        print("❌ Combined data file not found!")
        return
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Basic dataset overview
    print(f"📊 DATASET OVERVIEW:")
    print(f"   Total records: {len(df):,}")
    print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   Number of stocks: {df['Symbol'].nunique()}")
    print(f"   File size: {data_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Stock breakdown
    print(f"\n📈 STOCK BREAKDOWN:")
    print("-" * 50)
    
    ftse_stocks = []
    sp500_stocks = []
    
    for symbol in sorted(df['Symbol'].unique()):
        stock_data = df[df['Symbol'] == symbol]
        records = len(stock_data)
        start_date = stock_data['Date'].min().strftime('%Y-%m-%d')
        end_date = stock_data['Date'].max().strftime('%Y-%m-%d')
        
        if symbol.endswith('.L'):
            ftse_stocks.append((symbol, records, start_date, end_date))
        else:
            sp500_stocks.append((symbol, records, start_date, end_date))
    
    print("🇬🇧 FTSE 100 STOCKS:")
    for symbol, records, start, end in ftse_stocks:
        print(f"   {symbol:10s}: {records:,} records ({start} to {end})")
    
    print("\n🇺🇸 S&P 500 STOCKS:")
    for symbol, records, start, end in sp500_stocks:
        print(f"   {symbol:10s}: {records:,} records ({start} to {end})")
    
    # Performance analysis (NO MODEL TRAINING)
    print(f"\n📊 10-YEAR PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    performance_data = []
    
    for symbol in df['Symbol'].unique():
        stock_data = df[df['Symbol'] == symbol].sort_values('Date')
        
        if len(stock_data) > 1:
            start_price = stock_data['Close'].iloc[0]
            end_price = stock_data['Close'].iloc[-1]
            
            # Calculate total return
            total_return = ((end_price - start_price) / start_price) * 100
            
            # Calculate annualized volatility
            daily_returns = stock_data['Daily_Return'].dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Calculate max drawdown
            running_max = stock_data['Close'].expanding().max()
            drawdown = (stock_data['Close'] - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            performance_data.append({
                'Symbol': symbol,
                'Start_Price': start_price,
                'End_Price': end_price,
                'Total_Return_Pct': total_return,
                'Annualized_Volatility_Pct': volatility,
                'Max_Drawdown_Pct': max_drawdown
            })
    
    # Create performance summary
    perf_df = pd.DataFrame(performance_data)
    perf_df = perf_df.sort_values('Total_Return_Pct', ascending=False)
    
    print("📈 BEST PERFORMERS (by total return):")
    for _, row in perf_df.head(5).iterrows():
        print(f"   {row['Symbol']:10s}: {row['Total_Return_Pct']:+7.1f}% return, {row['Annualized_Volatility_Pct']:5.1f}% volatility")
    
    print("\n📉 WORST PERFORMERS (by total return):")
    for _, row in perf_df.tail(5).iterrows():
        print(f"   {row['Symbol']:10s}: {row['Total_Return_Pct']:+7.1f}% return, {row['Annualized_Volatility_Pct']:5.1f}% volatility")
    
    # Market comparison
    print(f"\n🏆 MARKET COMPARISON:")
    print("-" * 50)
    
    ftse_returns = perf_df[perf_df['Symbol'].str.endswith('.L')]['Total_Return_Pct']
    sp500_returns = perf_df[~perf_df['Symbol'].str.endswith('.L')]['Total_Return_Pct']
    
    print(f"FTSE 100 stocks average return: {ftse_returns.mean():+6.1f}%")
    print(f"S&P 500 stocks average return:  {sp500_returns.mean():+6.1f}%")
    print(f"Overall average return:         {perf_df['Total_Return_Pct'].mean():+6.1f}%")
    
    # Data quality summary
    print(f"\n✅ DATA QUALITY SUMMARY:")
    print("-" * 50)
    print(f"Complete datasets: {len(performance_data)}/{df['Symbol'].nunique()}")
    print(f"Average records per stock: {len(df) / df['Symbol'].nunique():.0f}")
    print(f"Data completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
    
    # Save performance summary
    summary_file = Path("data/historical_10years/performance_summary.csv")
    perf_df.to_csv(summary_file, index=False)
    print(f"\n💾 Performance summary saved to: {summary_file}")
    
    print(f"\n🎯 DATA COLLECTION COMPLETE!")
    print("✅ All data collected successfully - NO MODELS WERE TRAINED")
    print("📁 Data location: data/historical_10years/")
    print("📊 Ready for future analysis or model training if needed")

if __name__ == "__main__":
    analyze_collected_data() 