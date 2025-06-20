#!/usr/bin/env python3
"""
Fixed script to collect market indices data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def collect_market_indices():
    """Collect market indices data for benchmarking."""
    print("📈 COLLECTING MARKET INDICES DATA (FIXED)")
    print("=" * 45)
    
    os.makedirs('data/external', exist_ok=True)
    
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
            # Use Ticker object instead of download function
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            
            if not data.empty:
                # Reset index to make date a column
                data = data.reset_index()
                
                df = pd.DataFrame({
                    'date': data['Date'],
                    'close_price': data['Close'],
                    'volume': data['Volume'] if 'Volume' in data.columns else 0,
                    'high': data['High'],
                    'low': data['Low'], 
                    'open_price': data['Open'],
                    'ticker': ticker,
                    'name': name
                })
                
                all_indices_data.append(df)
                
                # Save individual index data
                filename = f'data/external/index_{ticker.replace("^", "").replace("=F", "")}.csv'
                df.to_csv(filename, index=False)
                print(f"💾 Saved: {filename} ({len(df)} records)")
                
        except Exception as e:
            print(f"❌ Error downloading {ticker}: {e}")
            # Try alternative approach
            try:
                print(f"🔄 Trying alternative method for {ticker}...")
                data = yf.download(ticker, start='2020-01-01', progress=False)
                
                if not data.empty and len(data.shape) == 2:
                    df = pd.DataFrame({
                        'date': data.index,
                        'close_price': data['Close'],
                        'volume': data['Volume'] if 'Volume' in data.columns else 0,
                        'high': data['High'],
                        'low': data['Low'], 
                        'open_price': data['Open'],
                        'ticker': ticker,
                        'name': name
                    })
                    
                    all_indices_data.append(df)
                    
                    filename = f'data/external/index_{ticker.replace("^", "").replace("=F", "")}.csv'
                    df.to_csv(filename, index=False)
                    print(f"💾 Saved (alternative): {filename} ({len(df)} records)")
                    
            except Exception as e2:
                print(f"❌ Alternative method also failed for {ticker}: {e2}")
    
    # Combine all indices data
    if all_indices_data:
        combined_indices = pd.concat(all_indices_data, ignore_index=True)
        combined_indices.to_csv('data/external/market_indices_combined.csv', index=False)
        print(f"💾 Combined indices data: data/external/market_indices_combined.csv ({len(combined_indices)} records)")
        return combined_indices
    else:
        print("❌ No market indices data collected")
        return pd.DataFrame()

if __name__ == "__main__":
    result = collect_market_indices()
    if not result.empty:
        print(f"\n✅ Market indices collection complete! Collected {len(result)} total records")
    else:
        print(f"\n⚠️ Market indices collection had issues, but other data sources are complete") 