#!/usr/bin/env python3
"""
Comprehensive data collection script for the Forecast Accuracy Assessment Model.
Collects all external and internal data sources as specified in the project plan.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_acquisition.financial_data import FinancialDataCollector
from data_acquisition.economic_data import EconomicDataCollector  
from data_acquisition.sentiment_data import SentimentDataCollector

def create_directories():
    """Create necessary data directories."""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'data/internal',
        'data/economic',
        'data/sentiment'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def collect_economic_data():
    """Collect external economic data sources."""
    print("\n🏦 COLLECTING ECONOMIC DATA (EXTERNAL)")
    print("=" * 50)
    
    collector = EconomicDataCollector()
    
    # Collect all economic data
    economic_data = collector.collect_all_economic_data()
    
    if economic_data:
        # Save individual sources
        for source, data in economic_data.items():
            if not data.empty:
                filename = f'data/external/economic_{source}.csv'
                data.to_csv(filename, index=False)
                print(f"💾 Saved: {filename} ({len(data)} records)")
        
        # Create combined economic dataset
        combined_economic = []
        for source, data in economic_data.items():
            if not data.empty:
                data['data_source'] = source
                combined_economic.append(data)
        
        if combined_economic:
            combined_df = pd.concat(combined_economic, ignore_index=True)
            combined_df.to_csv('data/external/economic_combined.csv', index=False)
            print(f"💾 Combined economic data: data/external/economic_combined.csv ({len(combined_df)} records)")
        
        return economic_data
    else:
        print("❌ No economic data collected")
        return {}

def collect_sentiment_data():
    """Collect external sentiment data sources."""
    print("\n📱 COLLECTING SENTIMENT DATA (EXTERNAL)")
    print("=" * 50)
    
    collector = SentimentDataCollector()
    
    # Collect all sentiment data
    sentiment_data = collector.collect_all_sentiment_data()
    
    if sentiment_data:
        # Save individual sources
        for source, data in sentiment_data.items():
            if not data.empty:
                filename = f'data/external/sentiment_{source}.csv'
                data.to_csv(filename, index=False)
                print(f"💾 Saved: {filename} ({len(data)} records)")
        
        # Create combined sentiment dataset
        combined_sentiment = []
        for source, data in sentiment_data.items():
            if not data.empty:
                data['data_source'] = source
                combined_sentiment.append(data)
        
        if combined_sentiment:
            combined_df = pd.concat(combined_sentiment, ignore_index=True)
            combined_df.to_csv('data/external/sentiment_combined.csv', index=False)
            print(f"💾 Combined sentiment data: data/external/sentiment_combined.csv ({len(combined_df)} records)")
        
        return sentiment_data
    else:
        print("❌ No sentiment data collected")
        return {}

def collect_financial_fundamentals():
    """Collect internal financial fundamentals data."""
    print("\n📊 COLLECTING FINANCIAL FUNDAMENTALS (INTERNAL)")
    print("=" * 50)
    
    collector = FinancialDataCollector()
    
    # Define our FTSE 100 and S&P 500 companies
    ftse_companies = [
        'AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRDA.L',  # Top performers
        'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L'   # Bottom performers
    ]
    
    sp500_companies = [
        'NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX',  # Top performers
        'WBA', 'INTC', 'PARA', 'PAYC', 'F'     # Bottom performers
    ]
    
    all_companies = ftse_companies + sp500_companies
    
    # Collect financial data for each company
    all_financial_data = []
    
    for ticker in all_companies:
        print(f"📈 Collecting fundamentals for {ticker}...")
        try:
            # Get company info
            company_info = collector.get_company_info(ticker)
            
            # Get financial metrics
            financial_data = collector.get_financial_metrics(ticker)
            
            if not financial_data.empty:
                # Add company info to the dataframe
                for key, value in company_info.items():
                    if key not in financial_data.columns:
                        financial_data[key] = value
                
                # Add market classification
                if ticker.endswith('.L'):
                    financial_data['market'] = 'FTSE100'
                    financial_data['category'] = 'top_performers' if ticker in ftse_companies[:5] else 'bottom_performers'
                else:
                    financial_data['market'] = 'SP500'
                    financial_data['category'] = 'top_performers' if ticker in sp500_companies[:5] else 'bottom_performers'
                
                all_financial_data.append(financial_data)
                
                # Save individual company data
                filename = f'data/internal/fundamentals_{ticker.replace(".", "_")}.csv'
                financial_data.to_csv(filename, index=False)
                print(f"💾 Saved: {filename} ({len(financial_data)} records)")
                
            else:
                print(f"❌ No data for {ticker}")
                
        except Exception as e:
            print(f"❌ Error collecting data for {ticker}: {e}")
            continue
        
        # Rate limiting
        time.sleep(0.5)
    
    # Combine all financial data
    if all_financial_data:
        combined_financials = pd.concat(all_financial_data, ignore_index=True)
        combined_financials.to_csv('data/internal/financial_fundamentals_combined.csv', index=False)
        print(f"💾 Combined financial fundamentals: data/internal/financial_fundamentals_combined.csv ({len(combined_financials)} records)")
        return combined_financials
    else:
        print("❌ No financial fundamentals collected")
        return pd.DataFrame()

def collect_market_indices():
    """Collect market indices data for benchmarking."""
    print("\n📈 COLLECTING MARKET INDICES DATA")
    print("=" * 40)
    
    import yfinance as yf
    
    indices = {
        '^FTSE': 'FTSE 100 Index',
        '^GSPC': 'S&P 500 Index',
        '^VIX': 'VIX Volatility Index',
        '^DJI': 'Dow Jones Industrial Average'
    }
    
    all_indices_data = []
    
    for ticker, name in indices.items():
        print(f"📊 Downloading {name} ({ticker})...")
        try:
            data = yf.download(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            
            if not data.empty:
                df = pd.DataFrame({
                    'date': data.index,
                    'close_price': data['Close'],
                    'volume': data['Volume'],
                    'high': data['High'],
                    'low': data['Low'],
                    'open_price': data['Open'],
                    'ticker': ticker,
                    'name': name
                })
                
                all_indices_data.append(df)
                
                # Save individual index data
                filename = f'data/external/index_{ticker.replace("^", "")}.csv'
                df.to_csv(filename, index=False)
                print(f"💾 Saved: {filename} ({len(df)} records)")
                
        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")
    
    # Combine all indices data
    if all_indices_data:
        combined_indices = pd.concat(all_indices_data, ignore_index=True)
        combined_indices.to_csv('data/external/market_indices_combined.csv', index=False)
        print(f"💾 Combined indices data: data/external/market_indices_combined.csv ({len(combined_indices)} records)")
        return combined_indices
    else:
        print("❌ No market indices data collected")
        return pd.DataFrame()

def create_data_summary():
    """Create a summary of all collected data."""
    print("\n📋 DATA COLLECTION SUMMARY")
    print("=" * 50)
    
    summary = {
        'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'datasets': []
    }
    
    # Check each data directory
    data_dirs = ['data/external', 'data/internal']
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            print(f"\n📁 {data_dir.upper()}:")
            for file in files:
                filepath = os.path.join(data_dir, file)
                try:
                    df = pd.read_csv(filepath)
                    file_info = {
                        'filename': file,
                        'location': data_dir,
                        'records': len(df),
                        'columns': len(df.columns),
                        'size_mb': round(os.path.getsize(filepath) / (1024*1024), 2)
                    }
                    summary['datasets'].append(file_info)
                    print(f"  ✅ {file}: {len(df)} records, {len(df.columns)} columns, {file_info['size_mb']} MB")
                except Exception as e:
                    print(f"  ❌ {file}: Error reading file - {e}")
    
    # Save summary
    import json
    with open('data/data_collection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Data summary saved: data/data_collection_summary.json")
    
    # Print overall statistics
    total_records = sum(dataset['records'] for dataset in summary['datasets'])
    total_size = sum(dataset['size_mb'] for dataset in summary['datasets'])
    
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"📊 Total datasets: {len(summary['datasets'])}")
    print(f"📈 Total records: {total_records:,}")
    print(f"💾 Total size: {total_size:.2f} MB")

def main():
    """Main function to collect all data sources."""
    print("🌟 COMPREHENSIVE DATA COLLECTION")
    print("Asset Classes: FTSE 100 & S&P 500")
    print("Data Sources: Stock Prices + Economic + Sentiment + Fundamentals")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Collect all data sources
    try:
        # 1. Economic data (external)
        economic_data = collect_economic_data()
        
        # 2. Sentiment data (external) 
        sentiment_data = collect_sentiment_data()
        
        # 3. Financial fundamentals (internal)
        financial_data = collect_financial_fundamentals()
        
        # 4. Market indices (external)
        indices_data = collect_market_indices()
        
        # 5. Create data summary
        create_data_summary()
        
        print("\n🎉 DATA COLLECTION COMPLETE!")
        print("=" * 60)
        print("✅ External Data: Economic indicators, sentiment, market indices")
        print("✅ Internal Data: Financial fundamentals, company metrics")
        print("✅ Stock Prices: FTSE 100 & S&P 500 companies")
        print("📁 All data saved in organized directory structure")
        print("📊 Summary available in: data/data_collection_summary.json")
        print("\n🌐 Your forecasting system now has comprehensive data!")
        print("🚀 Web application: http://localhost:5001")
        
    except Exception as e:
        print(f"\n❌ Error in data collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 