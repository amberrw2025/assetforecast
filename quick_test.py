#!/usr/bin/env python3
"""
Quick test script for the Forecast Accuracy Assessment Model Pipeline.
This script tests the basic functionality without running the full pipeline.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_basic_imports():
    """Test basic Python package imports."""
    print("🧪 Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import requests
        import plotly.graph_objects as go
        print("✅ All basic packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("🧪 Testing configuration...")
    
    try:
        from config import COMPANIES, FINANCIAL_METRICS, ECONOMIC_INDICATORS
        print(f"✅ Configuration loaded successfully")
        print(f"   - Markets: {list(COMPANIES.keys())}")
        print(f"   - Financial metrics: {len(FINANCIAL_METRICS)}")
        print(f"   - Economic indicators: {len(ECONOMIC_INDICATORS)}")
        return True
    except ImportError as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_financial_collector():
    """Test financial data collector."""
    print("🧪 Testing financial data collector...")
    
    try:
        from data_acquisition.financial_data import FinancialDataCollector
        collector = FinancialDataCollector()
        
        # Test with a simple ticker
        info = collector.get_company_info("AAPL")
        print(f"✅ Financial collector works - Company: {info['name']}")
        return True
    except Exception as e:
        print(f"❌ Financial collector error: {e}")
        return False

def test_economic_collector():
    """Test economic data collector."""
    print("🧪 Testing economic data collector...")
    
    try:
        from data_acquisition.economic_data import EconomicDataCollector
        collector = EconomicDataCollector()
        
        # Test FRED data (will use sample data)
        data = collector.get_fred_data("FEDFUNDS", "Federal Funds Rate")
        print(f"✅ Economic collector works - {len(data)} records")
        return True
    except Exception as e:
        print(f"❌ Economic collector error: {e}")
        return False

def test_data_cleaner():
    """Test data cleaner."""
    print("🧪 Testing data cleaner...")
    
    try:
        from data_processing.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        
        # Create sample data
        import pandas as pd
        import numpy as np
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(10) * 100 + 1000
        })
        
        # Test cleaning
        cleaned = cleaner.handle_missing_values(sample_data)
        print(f"✅ Data cleaner works - {len(cleaned)} records")
        return True
    except Exception as e:
        print(f"❌ Data cleaner error: {e}")
        return False

def test_visualizer():
    """Test visualizer."""
    print("🧪 Testing visualizer...")
    
    try:
        from utils.visualization import DataVisualizer
        visualizer = DataVisualizer()
        
        # Create sample data
        import pandas as pd
        import numpy as np
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(10) * 100 + 1000
        })
        
        # Test plotting
        fig = visualizer.plot_time_series(sample_data, y_columns=['value'])
        print("✅ Visualizer works - Plot created successfully")
        return True
    except Exception as e:
        print(f"❌ Visualizer error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("FORECAST MODEL PIPELINE - QUICK TEST")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_config),
        ("Financial Collector", test_financial_collector),
        ("Economic Collector", test_economic_collector),
        ("Data Cleaner", test_data_cleaner),
        ("Visualizer", test_visualizer),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("Run 'python3 run_pipeline.py' to start the full pipeline.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Some components may not work correctly.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 