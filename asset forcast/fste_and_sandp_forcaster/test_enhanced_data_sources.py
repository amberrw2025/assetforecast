#!/usr/bin/env python3
"""Test Enhanced Data Sources Implementation"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def test_all_enhanced_data_sources():
    print("🚀 ENHANCED DATA SOURCES TEST")
    print("=" * 50)
    
    # Test 1: FRED Economic Indicators
    print("\n📊 Testing FRED Economic Indicators...")
    try:
        from fredapi import Fred
        fred = Fred(api_key='f57a50634dba5f945b6cfbecc034a755')
        
        indicators = {
            'Fed Funds': 'FEDFUNDS',
            '10Y Treasury': 'GS10',
            'US CPI': 'CPIAUCSL',
            'GBP/USD': 'DEXUSUK'
        }
        
        fred_success = 0
        for name, series_id in indicators.items():
            try:
                data = fred.get_series(series_id, limit=2)
                if not data.empty:
                    print(f"✅ {name}: {data.iloc[-1]:.3f}")
                    fred_success += 1
                else:
                    print(f"⚠️ {name}: No data")
            except:
                print(f"❌ {name}: Failed")
        
        print(f"FRED Success: {fred_success}/{len(indicators)}")
        
    except ImportError:
        print("❌ fredapi not installed")
        fred_success = 0
    
    # Test 2: Sector ETFs
    print("\n📈 Testing Sector ETFs...")
    sectors = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Small Cap': 'IWM'
    }
    
    sector_success = 0
    for name, ticker in sectors.items():
        try:
            data = yf.download(ticker, period='2d', progress=False)
            if not data.empty:
                # Handle new yfinance multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = [col for col in data.columns if 'Close' in col[0]][0]
                    price = float(data[close_col].iloc[-1])
                else:
                    price = float(data['Close'].iloc[-1])
                print(f"✅ {name}: ${price:.2f}")
                sector_success += 1
            else:
                print(f"⚠️ {name}: No data")
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)[:30]}...")
    
    print(f"Sector ETF Success: {sector_success}/{len(sectors)}")
    
    # Test 3: Volatility Indices
    print("\n📊 Testing Volatility Indices...")
    vol_indices = {
        'VIX': '^VIX',
        'NASDAQ Vol': '^VXN'
    }
    
    vol_success = 0
    for name, ticker in vol_indices.items():
        try:
            data = yf.download(ticker, period='2d', progress=False)
            if not data.empty:
                # Handle new yfinance multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = [col for col in data.columns if 'Close' in col[0]][0]
                    level = float(data[close_col].iloc[-1])
                else:
                    level = float(data['Close'].iloc[-1])
                print(f"✅ {name}: {level:.2f}")
                vol_success += 1
            else:
                print(f"⚠️ {name}: No data")
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)[:30]}...")
    
    print(f"Volatility Success: {vol_success}/{len(vol_indices)}")
    
    # Test 4: Currency & Commodities
    print("\n💱 Testing FX/Commodities...")
    fx_commodities = {
        'GBP/USD': 'GBPUSD=X',
        'Gold': 'GC=F',
        'Oil': 'CL=F'
    }
    
    fx_success = 0
    for name, ticker in fx_commodities.items():
        try:
            data = yf.download(ticker, period='2d', progress=False)
            if not data.empty:
                # Handle new yfinance multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = [col for col in data.columns if 'Close' in col[0]][0]
                    price = float(data[close_col].iloc[-1])
                else:
                    price = float(data['Close'].iloc[-1])
                print(f"✅ {name}: {price:.4f}")
                fx_success += 1
            else:
                print(f"⚠️ {name}: No data")
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)[:30]}...")
    
    print(f"FX/Commodity Success: {fx_success}/{len(fx_commodities)}")
    
    # Summary
    total_success = fred_success + sector_success + vol_success + fx_success
    total_tests = len(indicators) + len(sectors) + len(vol_indices) + len(fx_commodities)
    success_rate = total_success / total_tests * 100
    
    print(f"\n🎯 OVERALL RESULTS")
    print("=" * 30)
    print(f"Total Success: {total_success}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("✅ READY FOR IMPLEMENTATION!")
    elif success_rate >= 60:
        print("⚠️ MOSTLY READY - Fix some issues")
    else:
        print("❌ NEEDS WORK - Multiple failures")
    
    return success_rate

if __name__ == "__main__":
    test_all_enhanced_data_sources() 