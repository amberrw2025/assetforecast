#!/usr/bin/env python3
"""
Historical Stock Data Collection Script
Collects 10 years of data for all project stocks - NO MODEL TRAINING
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import time
import json

class HistoricalDataCollector:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "historical_10years"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Stock lists from your project
        self.ftse_stocks = {
            'AZN.L': 'AstraZeneca',
            'LSEG.L': 'London Stock Exchange Group', 
            'RKT.L': 'Reckitt Benckiser',
            'OCDO.L': 'Ocado Group',
            'CRH.L': 'CRH plc',
            'BT-A.L': 'BT Group',
            'VOD.L': 'Vodafone Group',
            'SSE.L': 'SSE plc',
            'GLEN.L': 'Glencore',
            'TSCO.L': 'Tesco'
        }
        
        self.sp500_stocks = {
            'NVDA': 'NVIDIA Corporation',
            'TSLA': 'Tesla Inc',
            'MRNA': 'Moderna Inc',
            'ZM': 'Zoom Video Communications',
            'NFLX': 'Netflix Inc',
            'WBA': 'Walgreens Boots Alliance',
            'INTC': 'Intel Corporation',
            'PARA': 'Paramount Global',
            'PAYC': 'Paycom Software',
            'F': 'Ford Motor Company'
        }
        
        # Combine all stocks
        self.all_stocks = {**self.ftse_stocks, **self.sp500_stocks}
        
        # Date range for 10 years
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365*10)
        
    def collect_stock_data(self, symbol, name):
        """Collect comprehensive historical data for a single stock"""
        print(f"📈 Collecting data for {symbol} ({name})...")
        
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
            
            if hist_data.empty:
                print(f"❌ No data available for {symbol}")
                return None
            
            hist_data.reset_index(inplace=True)
            hist_data['Symbol'] = symbol
            hist_data['Company'] = name
            
            # Add technical indicators
            hist_data['Daily_Return'] = hist_data['Close'].pct_change()
            hist_data['MA_20'] = hist_data['Close'].rolling(20).mean()
            hist_data['MA_50'] = hist_data['Close'].rolling(50).mean()
            
            # Save individual file
            filename = f"{symbol.replace('.', '_')}_10year_data.csv"
            filepath = self.data_dir / filename
            hist_data.to_csv(filepath, index=False)
            
            print(f"✅ Saved {len(hist_data)} records for {symbol}")
            return hist_data
            
        except Exception as e:
            print(f"❌ Error collecting {symbol}: {str(e)}")
            return None
    
    def collect_all_data(self):
        """Collect data for all stocks"""
        print("🚀 Starting 10-Year Historical Data Collection")
        print("⚠️  NO MODELS WILL BE TRAINED - DATA COLLECTION ONLY")
        print("="*60)
        
        all_data = []
        successful = 0
        
        for symbol, name in self.all_stocks.items():
            data = self.collect_stock_data(symbol, name)
            if data is not None:
                all_data.append(data)
                successful += 1
            time.sleep(0.5)  # Be nice to the API
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_filepath = self.data_dir / "all_stocks_10year_combined.csv"
            combined_data.to_csv(combined_filepath, index=False)
            print(f"✅ Combined file saved with {len(combined_data):,} total records")
        
        print(f"📋 Successfully collected data for {successful}/{len(self.all_stocks)} stocks")
        return all_data

def main():
    print("🎯 10-YEAR HISTORICAL STOCK DATA COLLECTOR")
    collector = HistoricalDataCollector()
    collector.collect_all_data()
    print("🎉 Data collection complete!")

if __name__ == "__main__":
    main() 