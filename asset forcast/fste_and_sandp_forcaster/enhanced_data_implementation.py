"""
Enhanced Data Implementation - Practical Integration with Existing Pipeline
Step-by-step implementation of high-value data sources for improved forecasting
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Optional
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config import (
    SECTOR_ETFS, VOLATILITY_INDICES, CURRENCY_COMMODITIES, 
    BOND_INDICATORS, ECONOMIC_INDICATORS, DATA_SOURCES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataImplementation:
    """Implements enhanced data sources step by step."""
    
    def __init__(self):
        self.start_date = DATA_SOURCES["start_date"]
        self.end_date = DATA_SOURCES.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        
    def test_enhanced_economic_indicators(self) -> Dict[str, bool]:
        """Test the new FRED economic indicators."""
        print("🧪 Testing Enhanced Economic Indicators...")
        print("=" * 50)
        
        # Test FRED API connection
        try:
            from fredapi import Fred
            fred_api_key = os.getenv('FRED_API_KEY', 'f57a50634dba5f945b6cfbecc034a755')
            fred = Fred(api_key=fred_api_key)
            
            # Test a few key indicators
            test_indicators = {
                'treasury_10y': 'GS10',
                'inflation_us': 'CPIAUCSL', 
                'gbp_usd': 'DEXUSUK',
                'consumer_confidence': 'UMCSENT'
            }
            
            results = {}
            for name, series_id in test_indicators.items():
                try:
                    data = fred.get_series(series_id, limit=10)
                    if not data.empty:
                        latest_value = data.iloc[-1]
                        latest_date = data.index[-1]
                        print(f"✅ {name}: {latest_value:.2f} (as of {latest_date.date()})")
                        results[name] = True
                    else:
                        print(f"⚠️ {name}: No data received")
                        results[name] = False
                except Exception as e:
                    print(f"❌ {name}: Error - {str(e)}")
                    results[name] = False
            
            success_rate = sum(results.values()) / len(results) * 100
            print(f"\n📊 Economic Indicators Success Rate: {success_rate:.1f}%")
            return results
            
        except ImportError:
            print("❌ fredapi not installed. Run: pip install fredapi")
            return {}
        except Exception as e:
            print(f"❌ FRED API Error: {str(e)}")
            return {}
    
    def test_sector_etf_data(self) -> Dict[str, bool]:
        """Test sector ETF data collection."""
        print("\n🧪 Testing Sector ETF Data...")
        print("=" * 50)
        
        # Test key sector ETFs
        test_sectors = ['technology', 'financials', 'energy', 'small_cap']
        results = {}
        
        for sector in test_sectors:
            if sector in SECTOR_ETFS:
                ticker = SECTOR_ETFS[sector]['ticker']
                try:
                    data = yf.download(ticker, period='5d', progress=False)
                    if not data.empty:
                        latest_price = data['Close'].iloc[-1]
                        print(f"✅ {sector} ({ticker}): ${latest_price:.2f}")
                        results[sector] = True
                    else:
                        print(f"⚠️ {sector} ({ticker}): No data")
                        results[sector] = False
                except Exception as e:
                    print(f"❌ {sector} ({ticker}): Error - {str(e)}")
                    results[sector] = False
        
        success_rate = sum(results.values()) / len(results) * 100 if results else 0
        print(f"\n📊 Sector ETF Success Rate: {success_rate:.1f}%")
        return results
    
    def test_volatility_indices(self) -> Dict[str, bool]:
        """Test volatility indices data collection."""
        print("\n🧪 Testing Volatility Indices...")
        print("=" * 50)
        
        # Test key volatility indices
        test_vol_indices = ['vix', 'vxn', 'rvx']
        results = {}
        
        for vol_index in test_vol_indices:
            if vol_index in VOLATILITY_INDICES:
                ticker = VOLATILITY_INDICES[vol_index]['ticker']
                try:
                    data = yf.download(ticker, period='5d', progress=False)
                    if not data.empty:
                        latest_value = data['Close'].iloc[-1]
                        print(f"✅ {vol_index} ({ticker}): {latest_value:.2f}")
                        results[vol_index] = True
                    else:
                        print(f"⚠️ {vol_index} ({ticker}): No data")
                        results[vol_index] = False
                except Exception as e:
                    print(f"❌ {vol_index} ({ticker}): Error - {str(e)}")
                    results[vol_index] = False
        
        success_rate = sum(results.values()) / len(results) * 100 if results else 0
        print(f"\n📊 Volatility Indices Success Rate: {success_rate:.1f}%")
        return results
    
    def test_currency_commodity_data(self) -> Dict[str, bool]:
        """Test currency and commodity data collection."""
        print("\n🧪 Testing Currency & Commodity Data...")
        print("=" * 50)
        
        # Test critical FX and commodities (especially GBP/USD for FTSE)
        test_instruments = ['gbp_usd_spot', 'dxy', 'gold', 'oil_wti']
        results = {}
        
        for instrument in test_instruments:
            if instrument in CURRENCY_COMMODITIES:
                ticker = CURRENCY_COMMODITIES[instrument]['ticker']
                try:
                    data = yf.download(ticker, period='5d', progress=False)
                    if not data.empty:
                        latest_value = data['Close'].iloc[-1]
                        print(f"✅ {instrument} ({ticker}): {latest_value:.4f}")
                        results[instrument] = True
                    else:
                        print(f"⚠️ {instrument} ({ticker}): No data")
                        results[instrument] = False
                except Exception as e:
                    print(f"❌ {instrument} ({ticker}): Error - {str(e)}")
                    results[instrument] = False
        
        success_rate = sum(results.values()) / len(results) * 100 if results else 0
        print(f"\n📊 Currency/Commodity Success Rate: {success_rate:.1f}%")
        return results
    
    def demonstrate_enhanced_features(self):
        """Demonstrate how enhanced data improves feature set."""
        print("\n🔧 Demonstrating Enhanced Features...")
        print("=" * 50)
        
        # Get sample stock data (AAPL)
        try:
            stock_data = yf.download('AAPL', start='2023-01-01', end='2024-01-01', progress=False)
            base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            print(f"📈 Base AAPL features: {len(base_features)}")
            print(f"   {base_features}")
            
            # Get VIX data for market regime
            vix_data = yf.download('^VIX', start='2023-01-01', end='2024-01-01', progress=False)
            
            # Get technology sector for relative performance
            tech_sector = yf.download('XLK', start='2023-01-01', end='2024-01-01', progress=False)
            
            # Get GBP/USD for international context
            gbp_usd = yf.download('GBPUSD=X', start='2023-01-01', end='2024-01-01', progress=False)
            
            enhanced_features = base_features + [
                'VIX_Level', 'VIX_High_Vol_Regime', 'VIX_Low_Vol_Regime',
                'Tech_Sector_Performance', 'Tech_Sector_Momentum',
                'GBP_USD_Rate', 'GBP_USD_Change',
                'Market_Regime', 'Cross_Asset_Signal'
            ]
            
            print(f"🚀 Enhanced AAPL features: {len(enhanced_features)}")
            print(f"   Added {len(enhanced_features) - len(base_features)} new features!")
            
            # Show sample correlation
            if not stock_data.empty and not vix_data.empty:
                # Calculate correlation between AAPL returns and VIX
                aapl_returns = stock_data['Close'].pct_change()
                vix_levels = vix_data['Close']
                
                # Align dates and calculate correlation
                common_dates = aapl_returns.index.intersection(vix_levels.index)
                if len(common_dates) > 20:
                    correlation = aapl_returns[common_dates].corr(vix_levels[common_dates])
                    print(f"\n📊 AAPL-VIX Correlation: {correlation:.3f}")
                    print(f"   This shows VIX is {'negatively' if correlation < 0 else 'positively'} correlated with AAPL")
                    print(f"   VIX data would {'improve' if abs(correlation) > 0.1 else 'slightly help'} AAPL predictions")
            
        except Exception as e:
            print(f"❌ Error in demonstration: {str(e)}")
    
    def create_sample_enhanced_dataset(self, ticker: str = 'AAPL', 
                                     start_date: str = '2023-01-01',
                                     end_date: str = '2024-01-01') -> pd.DataFrame:
        """Create a sample enhanced dataset for one stock."""
        print(f"\n🔧 Creating Enhanced Dataset for {ticker}...")
        print("=" * 50)
        
        try:
            # Get base stock data
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if stock_data.empty:
                print(f"❌ No stock data for {ticker}")
                return pd.DataFrame()
            
            # Create base dataset
            enhanced_df = pd.DataFrame({
                'date': stock_data.index,
                'open': stock_data['Open'],
                'high': stock_data['High'],
                'low': stock_data['Low'],
                'close': stock_data['Close'],
                'volume': stock_data['Volume'],
                'ticker': ticker
            }).reset_index(drop=True)
            
            # Add basic technical indicators
            enhanced_df['return_1d'] = enhanced_df['close'].pct_change()
            enhanced_df['sma_20'] = enhanced_df['close'].rolling(20).mean()
            enhanced_df['volatility'] = enhanced_df['return_1d'].rolling(20).std()
            
            base_columns = enhanced_df.shape[1]
            
            # Add VIX data
            try:
                vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
                if not vix_data.empty:
                    vix_df = pd.DataFrame({
                        'date': vix_data.index,
                        'vix_level': vix_data['Close'],
                        'vix_high_vol': (vix_data['Close'] > 25).astype(int),
                        'vix_low_vol': (vix_data['Close'] < 15).astype(int)
                    }).reset_index(drop=True)
                    
                    enhanced_df = enhanced_df.merge(vix_df, on='date', how='left')
                    print("✅ Added VIX indicators")
            except Exception as e:
                print(f"⚠️ VIX data failed: {str(e)}")
            
            # Add sector performance (Technology for AAPL)
            try:
                sector_data = yf.download('XLK', start=start_date, end=end_date, progress=False)
                if not sector_data.empty:
                    sector_df = pd.DataFrame({
                        'date': sector_data.index,
                        'sector_close': sector_data['Close'],
                        'sector_return': sector_data['Close'].pct_change()
                    }).reset_index(drop=True)
                    
                    enhanced_df = enhanced_df.merge(sector_df, on='date', how='left')
                    print("✅ Added sector performance")
            except Exception as e:
                print(f"⚠️ Sector data failed: {str(e)}")
            
            # Add currency data (USD strength)
            try:
                dxy_data = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)
                if not dxy_data.empty:
                    dxy_df = pd.DataFrame({
                        'date': dxy_data.index,
                        'dxy_level': dxy_data['Close'],
                        'dxy_change': dxy_data['Close'].pct_change()
                    }).reset_index(drop=True)
                    
                    enhanced_df = enhanced_df.merge(dxy_df, on='date', how='left')
                    print("✅ Added currency indicators")
            except Exception as e:
                print(f"⚠️ Currency data failed: {str(e)}")
            
            final_columns = enhanced_df.shape[1]
            added_features = final_columns - base_columns
            
            print(f"\n📊 Enhanced Dataset Summary:")
            print(f"   Original features: {base_columns}")
            print(f"   Enhanced features: {final_columns}")
            print(f"   Added features: {added_features}")
            print(f"   Records: {len(enhanced_df)}")
            print(f"   Date range: {enhanced_df['date'].min().date()} to {enhanced_df['date'].max().date()}")
            
            return enhanced_df
            
        except Exception as e:
            print(f"❌ Error creating enhanced dataset: {str(e)}")
            return pd.DataFrame()
    
    def run_full_implementation_test(self):
        """Run comprehensive test of all enhanced data sources."""
        print("🚀 ENHANCED DATA IMPLEMENTATION TEST")
        print("=" * 60)
        
        all_results = {}
        
        # Test all data sources
        all_results['economic'] = self.test_enhanced_economic_indicators()
        all_results['sectors'] = self.test_sector_etf_data()
        all_results['volatility'] = self.test_volatility_indices()
        all_results['fx_commodities'] = self.test_currency_commodity_data()
        
        # Demonstrate enhanced features
        self.demonstrate_enhanced_features()
        
        # Create sample enhanced dataset
        sample_dataset = self.create_sample_enhanced_dataset()
        
        # Summary
        print("\n🎯 IMPLEMENTATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        total_successes = 0
        
        for category, results in all_results.items():
            if results:
                successes = sum(results.values())
                total = len(results)
                total_tests += total
                total_successes += successes
                print(f"{category.upper()}: {successes}/{total} successful ({successes/total*100:.1f}%)")
        
        overall_success = total_successes / total_tests * 100 if total_tests > 0 else 0
        print(f"\nOVERALL SUCCESS RATE: {overall_success:.1f}%")
        
        if overall_success > 70:
            print("✅ READY FOR IMPLEMENTATION - Most data sources working!")
        elif overall_success > 50:
            print("⚠️ PARTIALLY READY - Some data sources need attention")
        else:
            print("❌ NOT READY - Multiple data source issues need fixing")
        
        return all_results, sample_dataset


def main():
    """Run the enhanced data implementation test."""
    implementation = EnhancedDataImplementation()
    results, sample_data = implementation.run_full_implementation_test()
    
    # Save sample enhanced dataset
    if not sample_data.empty:
        output_file = "sample_enhanced_dataset.csv"
        sample_data.to_csv(output_file, index=False)
        print(f"\n💾 Saved sample enhanced dataset to: {output_file}")


if __name__ == "__main__":
    main() 