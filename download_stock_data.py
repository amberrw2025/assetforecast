#!/usr/bin/env python3
"""
Download real stock data for the forecasting web application.
Uses yfinance to get historical stock prices.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def download_stock_data(symbol='AAPL', period='2y', save_to_uploads=True):
    """
    Download stock data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        period (str): Time period ('1y', '2y', '5y', 'max')
        save_to_uploads (bool): Whether to save to uploads directory
    
    Returns:
        pd.DataFrame: Stock data with date and close_price columns
    """
    
    print(f"📈 Downloading {symbol} stock data for period: {period}")
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Prepare data for the web app
        df = pd.DataFrame({
            'date': data.index.date,
            'close_price': data['Close'].values
        })
        
        # Reset index and ensure proper data types
        df.reset_index(drop=True, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['close_price'] = df['close_price'].astype(float)
        
        # Remove any NaN values
        df = df.dropna()
        
        print(f"✅ Downloaded {len(df)} records")
        print(f"📅 Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"💰 Price range: ${df['close_price'].min():.2f} to ${df['close_price'].max():.2f}")
        print(f"📊 Latest price: ${df['close_price'].iloc[-1]:.2f}")
        
        if save_to_uploads:
            # Save to uploads directory
            filename = f'uploads/{symbol.lower()}_stock_data.csv'
            df.to_csv(filename, index=False)
            print(f"💾 Saved to: {filename}")
            
            # Also save as processed_data.csv for immediate use
            processed_filename = 'uploads/processed_data.csv'
            df.to_csv(processed_filename, index=False)
            print(f"💾 Also saved as: {processed_filename}")
            print("🚀 Data is now ready for the web application!")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def get_popular_stocks():
    """Return a list of popular stock symbols."""
    return {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corporation',
        'AMZN': 'Amazon.com Inc.',
        'TSLA': 'Tesla Inc.',
        'META': 'Meta Platforms Inc.',
        'NVDA': 'NVIDIA Corporation',
        'NFLX': 'Netflix Inc.',
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust'
    }

def download_multiple_stocks(symbols=None, period='1y'):
    """Download data for multiple stocks."""
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'TSLA']
    
    all_data = {}
    for symbol in symbols:
        print(f"\n{'='*50}")
        data = download_stock_data(symbol, period, save_to_uploads=False)
        if data is not None:
            all_data[symbol] = data
    
    return all_data

def main():
    """Main function with interactive stock selection."""
    print("🌟 Real Stock Data Downloader for Forecast Web App")
    print("=" * 60)
    
    # Show popular stocks
    stocks = get_popular_stocks()
    print("\n📊 Popular stocks available:")
    for symbol, name in stocks.items():
        print(f"  {symbol}: {name}")
    
    print("\n🔍 Options:")
    print("1. Download Apple (AAPL) - 2 years [DEFAULT]")
    print("2. Download Tesla (TSLA) - 2 years")
    print("3. Download S&P 500 ETF (SPY) - 2 years")
    print("4. Custom stock symbol")
    print("5. Download multiple stocks")
    
    try:
        choice = input("\nEnter your choice (1-5) or press Enter for default: ").strip()
        
        if choice == '' or choice == '1':
            # Default: Apple 2 years
            download_stock_data('AAPL', '2y')
            
        elif choice == '2':
            download_stock_data('TSLA', '2y')
            
        elif choice == '3':
            download_stock_data('SPY', '2y')
            
        elif choice == '4':
            symbol = input("Enter stock symbol (e.g., GOOGL): ").strip().upper()
            period = input("Enter period (1y, 2y, 5y, max) [2y]: ").strip() or '2y'
            download_stock_data(symbol, period)
            
        elif choice == '5':
            symbols_input = input("Enter symbols separated by commas (e.g., AAPL,GOOGL,TSLA): ").strip()
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
            period = input("Enter period (1y, 2y, 5y, max) [1y]: ").strip() or '1y'
            
            # Download the first one to uploads for immediate use
            if symbols:
                download_stock_data(symbols[0], period)
                if len(symbols) > 1:
                    download_multiple_stocks(symbols[1:], period)
        
        else:
            print("❌ Invalid choice. Downloading Apple (AAPL) as default.")
            download_stock_data('AAPL', '2y')
    
    except KeyboardInterrupt:
        print("\n\n👋 Download cancelled.")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("📈 Downloading Apple (AAPL) as fallback...")
        download_stock_data('AAPL', '2y')
    
    print("\n" + "=" * 60)
    print("🎉 Stock data download complete!")
    print("🌐 Your web application now has real stock data!")
    print("📱 Go to http://localhost:5001 to see the data and generate forecasts")
    print("=" * 60)

if __name__ == "__main__":
    main() 