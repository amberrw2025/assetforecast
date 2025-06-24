#!/usr/bin/env python3
"""
Download FTSE 100 and S&P 500 data for forecasting analysis.
Focus on top and bottom performers based on pre-2023 data as per the project plan.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

# Import start date from config
from config import DATA_SOURCES

def get_ftse100_companies():
    """
    Get FTSE 100 companies with their Yahoo Finance tickers.
    Based on historical performance analysis (pre-2023).
    """
    return {
        # Top 5 FTSE 100 performers (pre-2023)
        'top_performers': {
            'AZN.L': 'AstraZeneca PLC',
            'LSEG.L': 'London Stock Exchange Group',
            'RKT.L': 'Reckitt Benckiser Group',
            'OCDO.L': 'Ocado Group PLC',
            'CRDA.L': 'Croda International PLC'
        },
        # Bottom 5 FTSE 100 performers (pre-2023)
        'bottom_performers': {
            'BT-A.L': 'BT Group PLC',
            'VOD.L': 'Vodafone Group PLC',
            'SSE.L': 'SSE PLC',
            'GLEN.L': 'Glencore PLC',
            'TSCO.L': 'Tesco PLC'
        }
    }

def get_sp500_companies():
    """
    Get S&P 500 companies with their Yahoo Finance tickers.
    Based on historical performance analysis (pre-2023).
    """
    return {
        # Top 5 S&P 500 performers (pre-2023)
        'top_performers': {
            'NVDA': 'NVIDIA Corporation',
            'TSLA': 'Tesla Inc.',
            'MRNA': 'Moderna Inc.',
            'ZM': 'Zoom Video Communications',
            'NFLX': 'Netflix Inc.'
        },
        # Bottom 5 S&P 500 performers (pre-2023)
        'bottom_performers': {
            'WBA': 'Walgreens Boots Alliance',
            'INTC': 'Intel Corporation',
            'PARA': 'Paramount Global',
            'PAYC': 'Paycom Software Inc.',
            'F': 'Ford Motor Company'
        }
    }

def download_company_data(ticker, company_name):
    """
    Download stock data for a specific company.
    
    Args:
        ticker (str): Yahoo Finance ticker symbol
        company_name (str): Full company name
    
    Returns:
        pd.DataFrame: Stock data with date and close_price columns
    """
    print(f"📈 Downloading {company_name} ({ticker})...")
    
    try:
        start_date = DATA_SOURCES.get("start_date", "2020-01-01")
        
        # Download data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date)
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return None
        
        # Prepare data
        df = pd.DataFrame({
            'date': data.index.date,
            'close_price': data['Close'].values,
            'volume': data['Volume'].values,
            'high': data['High'].values,
            'low': data['Low'].values,
            'open': data['Open'].values
        })
        
        df.reset_index(drop=True, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna()
        
        # Add metadata
        df['ticker'] = ticker
        df['company_name'] = company_name
        
        print(f"✅ {ticker}: {len(df)} records ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return None

def calculate_performance_metrics(df, pre_2023_only=True):
    """Calculate performance metrics for a stock."""
    if df is None or df.empty:
        return None
    
    if pre_2023_only:
        # Filter to pre-2023 data for performance calculation
        df_filtered = df[df['date'] < '2023-01-01'].copy()
        if df_filtered.empty:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    if len(df_filtered) < 2:
        return None
    
    # Calculate metrics
    start_price = df_filtered['close_price'].iloc[0]
    end_price = df_filtered['close_price'].iloc[-1]
    
    total_return = ((end_price - start_price) / start_price) * 100
    volatility = df_filtered['close_price'].pct_change().std() * np.sqrt(252) * 100  # Annualized
    
    return {
        'total_return_pct': total_return,
        'volatility_pct': volatility,
        'start_price': start_price,
        'end_price': end_price,
        'max_price': df_filtered['close_price'].max(),
        'min_price': df_filtered['close_price'].min()
    }

def save_individual_files(all_data, index_name):
    """Save individual CSV files for each company."""
    base_dir = f'data/{index_name.lower()}'
    os.makedirs(base_dir, exist_ok=True)
    
    for category, companies in all_data.items():
        category_dir = f'{base_dir}/{category}'
        os.makedirs(category_dir, exist_ok=True)
        
        for ticker, df in companies.items():
            if df is not None:
                filename = f'{category_dir}/{ticker.replace(".", "_")}.csv'
                # Save only date and close_price for the web app
                df_simple = df[['date', 'close_price']].copy()
                df_simple.to_csv(filename, index=False)
                print(f"💾 Saved: {filename}")

def create_combined_dataset(ftse_data, sp500_data):
    """Create a combined dataset for analysis."""
    print("\n📊 Creating combined dataset...")
    
    combined_data = []
    
    # Process FTSE 100 data
    for category, companies in ftse_data.items():
        for ticker, df in companies.items():
            if df is not None:
                df_copy = df.copy()
                df_copy['index'] = 'FTSE100'
                df_copy['category'] = category
                combined_data.append(df_copy)
    
    # Process S&P 500 data
    for category, companies in sp500_data.items():
        for ticker, df in companies.items():
            if df is not None:
                df_copy = df.copy()
                df_copy['index'] = 'SP500'
                df_copy['category'] = category
                combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Save combined dataset
        os.makedirs('data', exist_ok=True)
        combined_df.to_csv('data/combined_ftse_sp500_data.csv', index=False)
        print(f"💾 Combined dataset saved: data/combined_ftse_sp500_data.csv")
        print(f"📊 Total records: {len(combined_df)}")
        
        return combined_df
    
    return None

def select_representative_sample(ftse_data, sp500_data):
    """Select one representative stock from each category for the web app."""
    print("\n🎯 Selecting representative samples for web application...")
    
    # Select best performing stocks from each category
    samples = {}
    
    # FTSE 100 top performer
    if ftse_data['top_performers']:
        best_ftse_top = max(ftse_data['top_performers'].items(), 
                           key=lambda x: len(x[1]) if x[1] is not None else 0)
        if best_ftse_top[1] is not None:
            samples['FTSE_Top'] = best_ftse_top
    
    # FTSE 100 bottom performer  
    if ftse_data['bottom_performers']:
        best_ftse_bottom = max(ftse_data['bottom_performers'].items(),
                              key=lambda x: len(x[1]) if x[1] is not None else 0)
        if best_ftse_bottom[1] is not None:
            samples['FTSE_Bottom'] = best_ftse_bottom
    
    # S&P 500 top performer
    if sp500_data['top_performers']:
        best_sp_top = max(sp500_data['top_performers'].items(),
                         key=lambda x: len(x[1]) if x[1] is not None else 0)
        if best_sp_top[1] is not None:
            samples['SP500_Top'] = best_sp_top
    
    # S&P 500 bottom performer
    if sp500_data['bottom_performers']:
        best_sp_bottom = max(sp500_data['bottom_performers'].items(),
                            key=lambda x: len(x[1]) if x[1] is not None else 0)
        if best_sp_bottom[1] is not None:
            samples['SP500_Bottom'] = best_sp_bottom
    
    return samples

def main():
    """Main function to download FTSE 100 and S&P 500 data."""
    print("🌟 FTSE 100 & S&P 500 Asset Class Data Downloader")
    print("=" * 60)
    print("📋 Plan: Top & Bottom 5 performers from each index (pre-2023 analysis)")
    print("=" * 60)
    
    # Get company lists
    ftse_companies = get_ftse100_companies()
    sp500_companies = get_sp500_companies()
    
    # Download FTSE 100 data
    print("\n🇬🇧 DOWNLOADING FTSE 100 DATA")
    print("-" * 40)
    
    ftse_data = {'top_performers': {}, 'bottom_performers': {}}
    
    for category, companies in ftse_companies.items():
        print(f"\n📊 {category.replace('_', ' ').title()}:")
        for ticker, name in companies.items():
            df = download_company_data(ticker, name)
            ftse_data[category][ticker] = df
            time.sleep(0.5)  # Be nice to the API
    
    # Download S&P 500 data
    print("\n🇺🇸 DOWNLOADING S&P 500 DATA")
    print("-" * 40)
    
    sp500_data = {'top_performers': {}, 'bottom_performers': {}}
    
    for category, companies in sp500_companies.items():
        print(f"\n📊 {category.replace('_', ' ').title()}:")
        for ticker, name in companies.items():
            df = download_company_data(ticker, name)
            sp500_data[category][ticker] = df
            time.sleep(0.5)  # Be nice to the API
    
    # Save individual files
    print("\n💾 SAVING DATA FILES")
    print("-" * 40)
    save_individual_files(ftse_data, 'FTSE100')
    save_individual_files(sp500_data, 'SP500')
    
    # Create combined dataset
    combined_df = create_combined_dataset(ftse_data, sp500_data)
    
    # Select representative samples for web app
    samples = select_representative_sample(ftse_data, sp500_data)
    
    # Save a representative sample for the web app
    if samples:
        print(f"\n🎯 REPRESENTATIVE SAMPLES FOR WEB APP")
        print("-" * 40)
        
        # Use the first available sample as the main dataset
        sample_name, (ticker, df) = next(iter(samples.items()))
        if df is not None:
            # Save as processed_data.csv for immediate use in web app
            df_simple = df[['date', 'close_price']].copy()
            df_simple.to_csv('uploads/processed_data.csv', index=False)
            print(f"🚀 Main dataset for web app: {ticker} ({sample_name})")
            print(f"📊 Records: {len(df_simple)}")
            print(f"📅 Date range: {df_simple['date'].min().strftime('%Y-%m-%d')} to {df_simple['date'].max().strftime('%Y-%m-%d')}")
        
        # Also save all samples individually
        for sample_name, (ticker, df) in samples.items():
            if df is not None:
                df_simple = df[['date', 'close_price']].copy()
                filename = f'uploads/{sample_name.lower()}_{ticker.replace(".", "_")}.csv'
                df_simple.to_csv(filename, index=False)
                print(f"💾 Sample saved: {filename}")
    
    # Print summary
    print(f"\n🎉 DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"📊 FTSE 100 companies downloaded: {sum(1 for cat in ftse_data.values() for df in cat.values() if df is not None)}/10")
    print(f"📊 S&P 500 companies downloaded: {sum(1 for cat in sp500_data.values() for df in cat.values() if df is not None)}/10")
    print(f"📁 Data saved in: data/ directory")
    print(f"🌐 Web app data ready: uploads/processed_data.csv")
    print(f"🚀 Go to http://localhost:5001 to analyze the data!")
    print("=" * 60)

if __name__ == "__main__":
    main() 