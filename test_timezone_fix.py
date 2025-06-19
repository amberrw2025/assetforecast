#!/usr/bin/env python3
"""
Test script to verify the timezone fix in the data cleaner.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def test_timezone_fix():
    """Test that timezone-aware datetime objects are handled correctly."""
    print("🧪 Testing timezone fix...")
    
    try:
        from data_processing.data_cleaner import DataCleaner
        
        # Create sample data with timezone-aware datetime
        dates = pd.date_range('2023-01-01', periods=10, freq='D', tz='UTC')
        sample_data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(10) * 100 + 1000
        })
        
        print(f"✅ Created sample data with timezone-aware dates: {len(sample_data)} records")
        
        # Test the data cleaner
        cleaner = DataCleaner()
        
        # Test time series standardization
        result = cleaner.standardize_time_series(sample_data)
        print(f"✅ Time series standardization works: {len(result)} records")
        
        # Test missing value handling
        result = cleaner.handle_missing_values(result)
        print(f"✅ Missing value handling works: {len(result)} records")
        
        # Test outlier detection
        result = cleaner.detect_and_handle_outliers(result)
        print(f"✅ Outlier detection works: {len(result)} records")
        
        print("🎉 All timezone-related tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Timezone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_step():
    """Test a step of the full pipeline with timezone-aware data."""
    print("\n🧪 Testing full pipeline step...")
    
    try:
        from data_acquisition.financial_data import FinancialDataCollector
        from data_processing.data_cleaner import DataCleaner
        
        # Get some real financial data (which has timezone-aware dates)
        collector = FinancialDataCollector()
        financial_data = collector.get_financial_metrics("AAPL")
        
        if not financial_data.empty:
            print(f"✅ Got financial data: {len(financial_data)} records")
            
            # Test cleaning this data
            cleaner = DataCleaner()
            
            # Test each cleaning step
            cleaned = cleaner.handle_missing_values(financial_data)
            print(f"✅ Missing values handled: {len(cleaned)} records")
            
            cleaned = cleaner.detect_and_handle_outliers(cleaned)
            print(f"✅ Outliers handled: {len(cleaned)} records")
            
            # This is the step that was failing
            cleaned = cleaner.standardize_time_series(cleaned)
            print(f"✅ Time series standardized: {len(cleaned)} records")
            
            print("🎉 Full pipeline step test passed!")
            return True
        else:
            print("⚠️  No financial data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Full pipeline step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all timezone tests."""
    print("="*60)
    print("TIMEZONE FIX TEST")
    print("="*60)
    
    # Test 1: Basic timezone fix
    test1_passed = test_timezone_fix()
    
    # Test 2: Full pipeline step
    test2_passed = test_full_pipeline_step()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("🎉 All timezone tests passed!")
        print("✅ The pipeline should now work correctly.")
        print("\nYou can now run:")
        print("  python3 run_pipeline.py")
    else:
        print("❌ Some timezone tests failed.")
        print("⚠️  The pipeline may still have issues.")
    
    print("="*60)
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 