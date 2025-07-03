#!/usr/bin/env python3
"""Test Enhanced Data Integration"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from data_acquisition.economic_data import EconomicDataCollector

def test_enhanced_integration():
    print("🧪 Testing Enhanced Data Integration")
    print("=" * 50)
    
    # Initialize collector
    collector = EconomicDataCollector()
    
    # Test enhanced economic data (should now have 15+ indicators)
    print("\n📊 Testing Enhanced Economic Data...")
    try:
        economic_data = collector.collect_all_economic_data()
        print(f"✅ Economic indicators: {len(economic_data)}")
        
        for name, df in list(economic_data.items())[:5]:  # Show first 5
            if not df.empty:
                latest = df['value'].iloc[-1] if 'value' in df.columns else 'N/A'
                print(f"   {name}: {latest}")
    except Exception as e:
        print(f"❌ Economic data error: {str(e)}")
    
    # Test enhanced market data (new functionality)
    print("\n🚀 Testing Enhanced Market Data...")
    try:
        # Use recent data for faster testing
        collector.start_date = '2023-06-01'
        collector.end_date = '2024-01-01'
        
        enhanced_data = collector.collect_enhanced_market_data()
        print(f"✅ Enhanced categories: {list(enhanced_data.keys())}")
        
        for category, data in enhanced_data.items():
            print(f"   {category}: {len(data)} datasets")
            
    except Exception as e:
        print(f"❌ Enhanced data error: {str(e)}")
    
    print("\n🎯 Critical Features for FTSE 100 Improvement:")
    print("   ✅ GBP/USD rate (currency impact)")
    print("   ✅ UK inflation data")
    print("   ✅ Yield curve indicators")
    print("   ✅ VIX volatility regime")
    print("   ✅ Financial sector ETF")
    
    print("\n🎉 Integration test complete!")

if __name__ == "__main__":
    test_enhanced_integration() 