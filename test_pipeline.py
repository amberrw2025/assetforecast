#!/usr/bin/env python3
"""
Test script for the Forecast Accuracy Assessment Model Pipeline.
This script tests individual components to ensure they work correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from data_acquisition import FinancialDataCollector, EconomicDataCollector, SentimentDataCollector
from data_processing import DataCleaner
from utils import get_logger, DataVisualizer

logger = get_logger("test_pipeline")

def test_financial_data_collector():
    """Test the financial data collector with sample data."""
    print("Testing Financial Data Collector...")
    
    collector = FinancialDataCollector()
    
    # Test with a single company
    test_ticker = "AAPL"
    
    try:
        # Test company info
        info = collector.get_company_info(test_ticker)
        assert info is not None and 'name' in info, "Company info should not be None"
        print(f"✓ Company info retrieved for {test_ticker}: {info['name']}")
        
        # Test financial metrics
        financial_data = collector.get_financial_metrics(test_ticker)
        assert not financial_data.empty, "Financial data should not be empty"
        print(f"✓ Financial data retrieved: {len(financial_data)} records")
        print(f"  Columns: {list(financial_data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing financial data collector: {str(e)}")
        return False

def test_economic_data_collector():
    """Test the economic data collector with sample data."""
    print("\nTesting Economic Data Collector...")
    
    collector = EconomicDataCollector()
    
    try:
        # Test FRED data (will use sample data if no API key)
        fred_data = collector.get_fred_data("FEDFUNDS", "Federal Funds Rate")
        assert not fred_data.empty, "FRED data should not be empty"
        print(f"✓ FRED data retrieved: {len(fred_data)} records")
        
        # Test EIA data (will use sample data)
        eia_data = collector.get_eia_data("RBRTE", "Brent Crude Oil Price")
        assert not eia_data.empty, "EIA data should not be empty"
        print(f"✓ EIA data retrieved: {len(eia_data)} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing economic data collector: {str(e)}")
        return False

def test_sentiment_data_collector():
    """Test the sentiment data collector with sample data."""
    print("\nTesting Sentiment Data Collector...")

    collector = SentimentDataCollector()
    collector._initialize_twitter()
    collector._initialize_reddit()

    try:
        # Test Twitter data (will use sample data if no API keys)
        keywords = ["inflation", "recession"]
        all_twitter_data = []
        for keyword in keywords:
            twitter_data = collector.get_tweets(keyword, limit=10)
            if not twitter_data.empty:
                all_twitter_data.append(twitter_data)

        if all_twitter_data:
            twitter_df = pd.concat(all_twitter_data)
            print(f"✓ Twitter data retrieved: {len(twitter_df)} records")
        else:
            print("⚠ No Twitter data retrieved")

        # Test Reddit data (will use sample data if no API keys)
        subreddits = ["investing", "stocks"]
        all_reddit_data = []
        for subreddit in subreddits:
            for keyword in keywords:
                reddit_data = collector.get_reddit_posts(subreddit, keyword, limit=10)
                if not reddit_data.empty:
                    all_reddit_data.append(reddit_data)
        
        if all_reddit_data:
            reddit_df = pd.concat(all_reddit_data)
            print(f"✓ Reddit data retrieved: {len(reddit_df)} records")
        else:
            print("⚠ No Reddit data retrieved")

        return True

    except Exception as e:
        print(f"✗ Error testing sentiment data collector: {str(e)}")
        return False

def test_data_cleaner():
    """Test the data cleaner with sample data."""
    print("\nTesting Data Cleaner...")
    
    cleaner = DataCleaner()
    
    try:
        # Create sample dataset
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'value1': np.random.randn(len(dates)) * 100 + 1000,
            'value2': np.random.randn(len(dates)) * 50 + 500,
            'category': np.random.choice(['A', 'B', 'C'], len(dates))
        })
        
        # Add some missing values
        sample_data.loc[10:15, 'value1'] = np.nan
        sample_data.loc[20:25, 'value2'] = np.nan
        
        print(f"✓ Sample dataset created: {sample_data.shape}")
        
        # Test missing value handling
        cleaned_data = cleaner.handle_missing_values(sample_data)
        print(f"✓ Missing values handled: {cleaned_data.isnull().sum().sum()} remaining")
        
        # Test outlier detection
        cleaned_data = cleaner.detect_and_handle_outliers(cleaned_data, ['value1', 'value2'])
        print(f"✓ Outliers handled")
        
        # Test time series standardization
        standardized_data = cleaner.standardize_time_series(cleaned_data)
        print(f"✓ Time series standardized: {standardized_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data cleaner: {str(e)}")
        return False

def test_visualizer():
    """Test the visualization utilities."""
    print("\nTesting Data Visualizer...")
    
    visualizer = DataVisualizer()
    
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'value1': np.random.randn(len(dates)) * 100 + 1000,
            'value2': np.random.randn(len(dates)) * 50 + 500,
            'category': np.random.choice(['A', 'B', 'C'], len(dates))
        })
        
        # Test time series plot
        fig = visualizer.plot_time_series(sample_data, y_columns=['value1', 'value2'])
        print("✓ Time series plot created")
        
        # Test correlation matrix
        fig = visualizer.plot_correlation_matrix(sample_data)
        print("✓ Correlation matrix created")
        
        # Test missing values plot
        fig = visualizer.plot_missing_values(sample_data)
        print("✓ Missing values plot created")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing visualizer: {str(e)}")
        return False

def test_configuration():
    """Test that configuration is loaded correctly."""
    print("\nTesting Configuration...")
    
    try:
        # Test that key configurations are available
        assert hasattr(COMPANIES, 'keys'), "COMPANIES configuration not found"
        assert hasattr(FINANCIAL_METRICS, '__iter__'), "FINANCIAL_METRICS configuration not found"
        assert hasattr(ECONOMIC_INDICATORS, 'keys'), "ECONOMIC_INDICATORS configuration not found"
        assert hasattr(CLEANING_CONFIG, 'keys'), "CLEANING_CONFIG configuration not found"
        
        print("✓ All configuration sections loaded correctly")
        print(f"  - Companies: {len(COMPANIES)} markets configured")
        print(f"  - Financial metrics: {len(FINANCIAL_METRICS)} metrics")
        print(f"  - Economic indicators: {len(ECONOMIC_INDICATORS)} indicators")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing configuration: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and provide a summary."""
    print("="*60)
    print("FORECAST MODEL PIPELINE - COMPONENT TESTS")
    print("="*60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Financial Data Collector", test_financial_data_collector),
        ("Economic Data Collector", test_economic_data_collector),
        ("Sentiment Data Collector", test_sentiment_data_collector),
        ("Data Cleaner", test_data_cleaner),
        ("Data Visualizer", test_visualizer),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Unexpected error in {test_name}: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("Run 'python run_pipeline.py' to start the full pipeline.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("Some components may not work correctly.")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 