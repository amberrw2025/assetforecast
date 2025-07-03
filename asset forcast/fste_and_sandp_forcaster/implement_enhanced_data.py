"""
Enhanced Data Sources Implementation
Practical example of adding high-value data sources to improve forecasting accuracy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import os
from typing import Dict, List, Optional
import logging

# Try to import FRED API
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️ Install fredapi with: pip install fredapi")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """Collects enhanced data sources for improved forecasting."""
    
    def __init__(self, fred_api_key: str = None):
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred = None
        
        if FRED_AVAILABLE and self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
                logger.info("✅ FRED API initialized")
            except Exception as e:
                logger.warning(f"⚠️ FRED API failed: {e}")
        else:
            logger.warning("⚠️ FRED API not available, using fallback data")
    
    def get_enhanced_economic_indicators(self, start_date: str = '2015-01-01', 
                                       end_date: str = None) -> pd.DataFrame:
        """
        Get enhanced economic indicators that significantly improve predictions.
        
        Returns:
            DataFrame with economic indicators
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        indicators = {}
        
        # High-impact economic indicators
        fred_series = {
            'inflation_us': 'CPIAUCSL',           # US CPI (inflation)
            'inflation_uk': 'GBRCPIALLMINMEI',   # UK CPI
            'treasury_3m': 'GS3M',               # 3-Month Treasury
            'treasury_2y': 'GS2',                # 2-Year Treasury  
            'treasury_5y': 'GS5',                # 5-Year Treasury
            'treasury_30y': 'GS30',              # 30-Year Treasury
            'consumer_confidence': 'UMCSENT',     # Consumer Sentiment
            'industrial_production': 'INDPRO',   # Industrial Production
            'credit_spread': 'BAMLC0A0CM',       # Investment Grade Spread
            'high_yield_spread': 'BAMLH0A0HYM2', # High Yield Spread
            'gbp_usd': 'DEXUSUK',                # GBP/USD Exchange Rate
        }
        
        all_data = []
        
        for name, series_id in fred_series.items():
            data = self._get_fred_series(series_id, name, start_date, end_date)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            # Pivot to wide format
            pivot_data = result.pivot(index='date', columns='indicator', values='value')
            
            # Calculate derived indicators
            if 'treasury_10y' in pivot_data.columns and 'treasury_3m' in pivot_data.columns:
                pivot_data['yield_curve_spread'] = pivot_data['treasury_10y'] - pivot_data['treasury_3m']
            
            return pivot_data.reset_index()
        
        return pd.DataFrame()
    
    def get_sector_performance_data(self, start_date: str = '2015-01-01',
                                  end_date: str = None) -> pd.DataFrame:
        """
        Get sector ETF performance for relative analysis.
        
        Returns:
            DataFrame with sector performance data
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Key sector ETFs
        sector_etfs = {
            'technology': 'XLK',
            'financials': 'XLF', 
            'healthcare': 'XLV',
            'energy': 'XLE',
            'utilities': 'XLU',
            'consumer_discretionary': 'XLY',
            'consumer_staples': 'XLP',
            'industrials': 'XLI',
            'materials': 'XLB',
            'real_estate': 'XLRE',
            'small_cap': 'IWM',
            'emerging_markets': 'EEM',
        }
        
        sector_data = {}
        
        for sector_name, ticker in sector_etfs.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    # Calculate returns and momentum
                    sector_data[f'{sector_name}_close'] = data['Close']
                    sector_data[f'{sector_name}_return_1d'] = data['Close'].pct_change()
                    sector_data[f'{sector_name}_return_5d'] = data['Close'].pct_change(5)
                    sector_data[f'{sector_name}_return_20d'] = data['Close'].pct_change(20)
                    
                logger.info(f"✅ Collected {sector_name} sector data")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get {sector_name} data: {e}")
        
        if sector_data:
            df = pd.DataFrame(sector_data)
            df.index.name = 'date'
            return df.reset_index()
        
        return pd.DataFrame()
    
    def get_volatility_indicators(self, start_date: str = '2015-01-01',
                                end_date: str = None) -> pd.DataFrame:
        """
        Get comprehensive volatility indicators.
        
        Returns:
            DataFrame with volatility indicators
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        volatility_tickers = {
            'vix': '^VIX',           # S&P 500 volatility
            'vix9d': '^VIX9D',       # 9-day VIX
            'vxn': '^VXN',           # NASDAQ volatility
            'rvx': '^RVX',           # Russell 2000 volatility
            'move': '^MOVE',         # Bond volatility index
        }
        
        vol_data = {}
        
        for name, ticker in volatility_tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    vol_data[f'{name}_close'] = data['Close']
                    vol_data[f'{name}_change'] = data['Close'].pct_change()
                    vol_data[f'{name}_ma20'] = data['Close'].rolling(20).mean()
                    
                logger.info(f"✅ Collected {name} volatility data")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get {name} data: {e}")
        
        if vol_data:
            df = pd.DataFrame(vol_data)
            df.index.name = 'date'
            return df.reset_index()
        
        return pd.DataFrame()
    
    def get_currency_and_commodity_data(self, start_date: str = '2015-01-01',
                                      end_date: str = None) -> pd.DataFrame:
        """
        Get currency and commodity data crucial for international stocks.
        
        Returns:
            DataFrame with currency and commodity data
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Important currency and commodity tickers
        tickers = {
            'dxy': 'DX-Y.NYB',       # Dollar Index
            'gbp_usd': 'GBPUSD=X',   # GBP/USD
            'eur_usd': 'EURUSD=X',   # EUR/USD
            'gold': 'GC=F',          # Gold futures
            'oil_wti': 'CL=F',       # WTI Crude Oil
            'oil_brent': 'BZ=F',     # Brent Crude Oil
            'copper': 'HG=F',        # Copper futures
            'silver': 'SI=F',        # Silver futures
        }
        
        fx_commodity_data = {}
        
        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    fx_commodity_data[f'{name}_close'] = data['Close']
                    fx_commodity_data[f'{name}_return'] = data['Close'].pct_change()
                    fx_commodity_data[f'{name}_volatility'] = data['Close'].pct_change().rolling(20).std()
                    
                logger.info(f"✅ Collected {name} data")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get {name} data: {e}")
        
        if fx_commodity_data:
            df = pd.DataFrame(fx_commodity_data)
            df.index.name = 'date'
            return df.reset_index()
        
        return pd.DataFrame()
    
    def _get_fred_series(self, series_id: str, name: str, start_date: str, 
                        end_date: str) -> pd.DataFrame:
        """Get a single FRED series."""
        try:
            if self.fred:
                data = self.fred.get_series(series_id, start=start_date, end=end_date)
                if not data.empty:
                    return pd.DataFrame({
                        'date': data.index,
                        'value': data.values,
                        'indicator': name
                    })
        except Exception as e:
            logger.warning(f"⚠️ Failed to get FRED series {series_id}: {e}")
        
        return pd.DataFrame()
    
    def collect_all_enhanced_data(self, start_date: str = '2015-01-01',
                                end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect all enhanced data sources.
        
        Returns:
            Dictionary containing all enhanced datasets
        """
        logger.info("🚀 Starting enhanced data collection...")
        
        enhanced_data = {}
        
        # Collect economic indicators
        economic_data = self.get_enhanced_economic_indicators(start_date, end_date)
        if not economic_data.empty:
            enhanced_data['economic'] = economic_data
            logger.info(f"✅ Economic data: {len(economic_data)} records")
        
        # Collect sector performance
        sector_data = self.get_sector_performance_data(start_date, end_date)
        if not sector_data.empty:
            enhanced_data['sectors'] = sector_data
            logger.info(f"✅ Sector data: {len(sector_data)} records")
        
        # Collect volatility indicators
        volatility_data = self.get_volatility_indicators(start_date, end_date)
        if not volatility_data.empty:
            enhanced_data['volatility'] = volatility_data
            logger.info(f"✅ Volatility data: {len(volatility_data)} records")
        
        # Collect currency and commodity data
        fx_commodity_data = self.get_currency_and_commodity_data(start_date, end_date)
        if not fx_commodity_data.empty:
            enhanced_data['fx_commodities'] = fx_commodity_data
            logger.info(f"✅ FX/Commodity data: {len(fx_commodity_data)} records")
        
        logger.info(f"🎉 Enhanced data collection complete! Collected {len(enhanced_data)} datasets")
        return enhanced_data


def create_enhanced_features(price_data: pd.DataFrame, enhanced_data: Dict[str, pd.DataFrame],
                           ticker: str) -> pd.DataFrame:
    """
    Create enhanced features by combining price data with external data sources.
    
    Args:
        price_data: Stock price data with OHLCV
        enhanced_data: Dictionary of enhanced datasets
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with enhanced features
    """
    logger.info(f"🔧 Creating enhanced features for {ticker}...")
    
    # Start with price data
    features = price_data.copy()
    
    # Merge with enhanced data sources
    for data_type, data in enhanced_data.items():
        if not data.empty and 'date' in data.columns:
            # Align dates
            data['date'] = pd.to_datetime(data['date'])
            features['date'] = pd.to_datetime(features['date'])
            
            # Merge on date
            features = features.merge(data, on='date', how='left')
            logger.info(f"✅ Merged {data_type} data")
    
    # Create regime indicators
    if 'yield_curve_spread' in features.columns:
        features['recession_signal'] = (features['yield_curve_spread'] < 0).astype(int)
    
    if 'vix_close' in features.columns:
        features['high_volatility_regime'] = (features['vix_close'] > 25).astype(int)
        features['low_volatility_regime'] = (features['vix_close'] < 15).astype(int)
    
    # Currency impact for international stocks
    if ticker.endswith('.L') and 'gbp_usd_close' in features.columns:
        # For UK stocks, GBP/USD is crucial
        features['currency_impact'] = features['gbp_usd_return']
        logger.info("✅ Added GBP/USD impact for UK stock")
    
    # Sector relative performance
    if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'] and 'technology_return_20d' in features.columns:
        features['sector_momentum'] = features['technology_return_20d']
    elif ticker in ['JPM', 'BAC', 'C', 'WFC'] and 'financials_return_20d' in features.columns:
        features['sector_momentum'] = features['financials_return_20d']
    
    logger.info(f"🎉 Enhanced features created: {features.shape[1]} total features")
    return features


def demonstrate_enhanced_data_impact():
    """Demonstrate the impact of enhanced data sources."""
    logger.info("🚀 Demonstrating Enhanced Data Sources Impact")
    
    # Initialize collector
    collector = EnhancedDataCollector()
    
    # Collect enhanced data
    enhanced_data = collector.collect_all_enhanced_data(
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    # Show what we collected
    print("\n📊 Enhanced Data Sources Collected:")
    print("=" * 50)
    
    for data_type, data in enhanced_data.items():
        print(f"\n{data_type.upper()}:")
        print(f"  Records: {len(data)}")
        print(f"  Columns: {list(data.columns)[:5]}...")  # Show first 5 columns
        if not data.empty:
            print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Example: Get stock data and merge with enhanced features
    print(f"\n🔧 Example: Enhanced Features for AAPL")
    print("=" * 50)
    
    # Get AAPL data
    aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01', progress=False)
    aapl_df = pd.DataFrame({
        'date': aapl_data.index,
        'close': aapl_data['Close'],
        'volume': aapl_data['Volume'],
        'high': aapl_data['High'],
        'low': aapl_data['Low'],
    }).reset_index(drop=True)
    
    # Create enhanced features
    enhanced_features = create_enhanced_features(aapl_df, enhanced_data, 'AAPL')
    
    print(f"Original features: {aapl_df.shape[1]}")
    print(f"Enhanced features: {enhanced_features.shape[1]}")
    print(f"Added features: {enhanced_features.shape[1] - aapl_df.shape[1]}")
    
    # Show sample enhanced features
    print(f"\nSample enhanced features for AAPL:")
    relevant_cols = [col for col in enhanced_features.columns 
                    if any(keyword in col.lower() for keyword in 
                          ['vix', 'yield', 'technology', 'recession', 'volatility'])][:10]
    if relevant_cols:
        print(enhanced_features[['date'] + relevant_cols].tail())


if __name__ == "__main__":
    demonstrate_enhanced_data_impact() 