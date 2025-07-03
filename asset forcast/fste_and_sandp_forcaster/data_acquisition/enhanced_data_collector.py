"""
Enhanced Data Collector for Improved Forecasting Accuracy
Extends existing data collection with sector ETFs, volatility indices, currency/commodities, and bonds
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import logging

from config import (
    SECTOR_ETFS, VOLATILITY_INDICES, CURRENCY_COMMODITIES, 
    BOND_INDICATORS, DATA_SOURCES
)
from utils.logger import get_logger
from .economic_data import EconomicDataCollector

logger = get_logger("enhanced_data_collector")

class EnhancedDataCollector(EconomicDataCollector):
    """
    Enhanced data collector that extends EconomicDataCollector with:
    - Sector ETF performance data
    - Volatility indices 
    - Currency and commodity data
    - Bond market indicators
    """
    
    def __init__(self):
        super().__init__()
        self.enhanced_data_cache = {}
        
    def get_yfinance_data(self, ticker_dict: Dict, start_date: str = None, 
                         end_date: str = None, data_type: str = "generic") -> Dict[str, pd.DataFrame]:
        """
        Generic method to fetch data from Yahoo Finance for multiple tickers.
        
        Args:
            ticker_dict: Dictionary with ticker info {name: {"ticker": "AAPL", "description": "..."}}
            start_date: Start date for data collection
            end_date: End date for data collection
            data_type: Type of data being collected (for logging)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of dataframes for each ticker
        """
        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date
            
        collected_data = {}
        
        logger.info(f"🚀 Collecting {data_type} data for {len(ticker_dict)} tickers...")
        
        for name, info in tqdm(ticker_dict.items(), desc=f"Fetching {data_type} data"):
            ticker = info["ticker"]
            description = info["description"]
            
            try:
                # Download data from Yahoo Finance
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty and len(data) > 0:
                    # Create standardized DataFrame
                    df = pd.DataFrame({
                        'date': data.index,
                        'open': data['Open'],
                        'high': data['High'], 
                        'low': data['Low'],
                        'close': data['Close'],
                        'volume': data['Volume'],
                        'ticker': ticker,
                        'name': name,
                        'description': description,
                        'source': 'Yahoo Finance',
                        'data_type': data_type
                    }).reset_index(drop=True)
                    
                    # Calculate additional features
                    df['return_1d'] = df['close'].pct_change()
                    df['return_5d'] = df['close'].pct_change(5)
                    df['return_20d'] = df['close'].pct_change(20)
                    df['volatility_20d'] = df['return_1d'].rolling(20).std()
                    df['sma_20'] = df['close'].rolling(20).mean()
                    df['sma_50'] = df['close'].rolling(50).mean()
                    
                    collected_data[name] = df
                    logger.info(f"✅ Collected {name} ({ticker}): {len(df)} records")
                    
                else:
                    logger.warning(f"⚠️ No data received for {name} ({ticker})")
                    
            except Exception as e:
                logger.error(f"❌ Error collecting {name} ({ticker}): {str(e)}")
                
            # Small delay to be respectful to Yahoo Finance
            time.sleep(0.1)
        
        logger.info(f"🎉 {data_type} collection complete: {len(collected_data)}/{len(ticker_dict)} successful")
        return collected_data
    
    def get_sector_etf_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect sector ETF performance data for relative analysis.
        
        Returns:
            Dict[str, pd.DataFrame]: Sector ETF data
        """
        logger.info("📊 Collecting Sector ETF Data...")
        return self.get_yfinance_data(SECTOR_ETFS, start_date, end_date, "sector_etf")
    
    def get_volatility_indices_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect volatility indices for market regime detection.
        
        Returns:
            Dict[str, pd.DataFrame]: Volatility indices data
        """
        logger.info("📈 Collecting Volatility Indices Data...")
        return self.get_yfinance_data(VOLATILITY_INDICES, start_date, end_date, "volatility_index")
    
    def get_currency_commodity_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect currency and commodity data.
        
        Returns:
            Dict[str, pd.DataFrame]: Currency and commodity data
        """
        logger.info("💱 Collecting Currency & Commodity Data...")
        return self.get_yfinance_data(CURRENCY_COMMODITIES, start_date, end_date, "currency_commodity")
    
    def get_bond_indicators_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect bond market indicators.
        
        Returns:
            Dict[str, pd.DataFrame]: Bond indicators data
        """
        logger.info("🏦 Collecting Bond Market Data...")
        return self.get_yfinance_data(BOND_INDICATORS, start_date, end_date, "bond_indicator")
    
    def create_derived_indicators(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Create derived indicators from the collected data.
        
        Args:
            data_dict: Dictionary containing all collected data categories
            
        Returns:
            pd.DataFrame: Derived indicators dataset
        """
        logger.info("🔧 Creating derived economic indicators...")
        
        derived_data = []
        
        # Get date range for alignment
        all_dates = set()
        for category_data in data_dict.values():
            for df in category_data.values():
                if not df.empty and 'date' in df.columns:
                    all_dates.update(df['date'].dt.date)
        
        if not all_dates:
            return pd.DataFrame()
            
        date_range = pd.DataFrame({'date': sorted(all_dates)})
        date_range['date'] = pd.to_datetime(date_range['date'])
        
        result_df = date_range.copy()
        
        # Yield Curve Indicators (if we have economic data with treasury rates)
        if 'economic' in data_dict:
            econ_data = data_dict['economic']
            
            # Try to find treasury rate data and create yield curve spreads
            treasury_data = {}
            for name, df in econ_data.items():
                if 'treasury' in name and not df.empty:
                    # Merge treasury data
                    df_clean = df[['date', 'value']].copy()
                    df_clean.columns = ['date', f'{name}_rate']
                    result_df = result_df.merge(df_clean, on='date', how='left')
                    treasury_data[name] = df_clean
            
            # Create yield curve spreads if we have the data
            if 'treasury_10y_rate' in result_df.columns and 'treasury_3m_rate' in result_df.columns:
                result_df['yield_curve_spread'] = result_df['treasury_10y_rate'] - result_df['treasury_3m_rate']
                result_df['yield_curve_inverted'] = (result_df['yield_curve_spread'] < 0).astype(int)
        
        # Volatility Regime Indicators
        if 'volatility' in data_dict:
            vol_data = data_dict['volatility']
            
            for name, df in vol_data.items():
                if not df.empty and 'vix' in name.lower():
                    df_clean = df[['date', 'close']].copy()
                    df_clean.columns = ['date', f'{name}_level']
                    result_df = result_df.merge(df_clean, on='date', how='left')
                    
                    # Create regime indicators
                    result_df[f'{name}_high_vol'] = (result_df[f'{name}_level'] > 25).astype(int)
                    result_df[f'{name}_low_vol'] = (result_df[f'{name}_level'] < 15).astype(int)
        
        # Currency Impact Indicators (critical for international stocks)
        if 'currency_commodity' in data_dict:
            fx_data = data_dict['currency_commodity']
            
            for name, df in vol_data.items():
                if 'gbp_usd' in name or 'eur_usd' in name or 'dxy' in name:
                    if not df.empty:
                        df_clean = df[['date', 'close', 'return_1d', 'return_20d']].copy()
                        df_clean.columns = ['date', f'{name}_rate', f'{name}_return_1d', f'{name}_return_20d']
                        result_df = result_df.merge(df_clean, on='date', how='left')
        
        # Sector Momentum Indicators
        if 'sector_etf' in data_dict:
            sector_data = data_dict['sector_etf']
            
            # Calculate relative sector performance
            if 'technology' in sector_data and 'financials' in sector_data:
                tech_df = sector_data['technology'][['date', 'return_20d']].copy()
                fin_df = sector_data['financials'][['date', 'return_20d']].copy()
                
                tech_df.columns = ['date', 'tech_return_20d']
                fin_df.columns = ['date', 'fin_return_20d']
                
                result_df = result_df.merge(tech_df, on='date', how='left')
                result_df = result_df.merge(fin_df, on='date', how='left')
                
                # Tech vs Financials momentum
                result_df['tech_vs_fin_momentum'] = result_df['tech_return_20d'] - result_df['fin_return_20d']
        
        logger.info(f"✅ Created derived indicators: {result_df.shape[1]} total features")
        return result_df
    
    def collect_all_enhanced_data(self, start_date: str = None, end_date: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect all enhanced data sources.
        
        Returns:
            Dict containing all enhanced datasets organized by category
        """
        logger.info("🚀 Starting comprehensive enhanced data collection...")
        
        if not start_date:
            start_date = self.start_date
        if not end_date:
            end_date = self.end_date
            
        all_enhanced_data = {}
        
        # Collect existing economic data (FRED)
        try:
            economic_data = self.collect_all_economic_data()
            if economic_data:
                all_enhanced_data['economic'] = economic_data
                total_econ_records = sum(len(df) for df in economic_data.values())
                logger.info(f"✅ Economic data: {total_econ_records} records across {len(economic_data)} indicators")
        except Exception as e:
            logger.error(f"❌ Error collecting economic data: {str(e)}")
        
        # Collect sector ETF data
        try:
            sector_data = self.get_sector_etf_data(start_date, end_date)
            if sector_data:
                all_enhanced_data['sector_etf'] = sector_data
                logger.info(f"✅ Sector ETF data: {len(sector_data)} sectors collected")
        except Exception as e:
            logger.error(f"❌ Error collecting sector data: {str(e)}")
        
        # Collect volatility indices
        try:
            volatility_data = self.get_volatility_indices_data(start_date, end_date)
            if volatility_data:
                all_enhanced_data['volatility'] = volatility_data
                logger.info(f"✅ Volatility data: {len(volatility_data)} indices collected")
        except Exception as e:
            logger.error(f"❌ Error collecting volatility data: {str(e)}")
        
        # Collect currency and commodity data
        try:
            fx_commodity_data = self.get_currency_commodity_data(start_date, end_date)
            if fx_commodity_data:
                all_enhanced_data['currency_commodity'] = fx_commodity_data
                logger.info(f"✅ FX/Commodity data: {len(fx_commodity_data)} instruments collected")
        except Exception as e:
            logger.error(f"❌ Error collecting FX/commodity data: {str(e)}")
        
        # Collect bond indicators
        try:
            bond_data = self.get_bond_indicators_data(start_date, end_date)
            if bond_data:
                all_enhanced_data['bond_indicators'] = bond_data
                logger.info(f"✅ Bond data: {len(bond_data)} instruments collected")
        except Exception as e:
            logger.error(f"❌ Error collecting bond data: {str(e)}")
        
        # Create derived indicators
        try:
            derived_indicators = self.create_derived_indicators(all_enhanced_data)
            if not derived_indicators.empty:
                all_enhanced_data['derived'] = {'market_indicators': derived_indicators}
                logger.info(f"✅ Derived indicators: {derived_indicators.shape[1]} features created")
        except Exception as e:
            logger.error(f"❌ Error creating derived indicators: {str(e)}")
        
        logger.info(f"🎉 Enhanced data collection complete! Categories: {list(all_enhanced_data.keys())}")
        return all_enhanced_data
    
    def save_enhanced_data(self, enhanced_data: Dict[str, Dict[str, pd.DataFrame]], 
                          output_dir: str = "data/raw"):
        """
        Save enhanced data to CSV files organized by category.
        
        Args:
            enhanced_data: Dictionary of enhanced datasets
            output_dir: Directory to save files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for category, category_data in enhanced_data.items():
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for name, df in category_data.items():
                if not df.empty:
                    filename = f"{name}.csv"
                    filepath = os.path.join(category_dir, filename)
                    df.to_csv(filepath, index=False)
                    logger.info(f"💾 Saved {category}/{name}: {len(df)} records to {filepath}")


def main():
    """Demonstrate enhanced data collection."""
    logger.info("🚀 Enhanced Data Collection Demo")
    
    # Initialize enhanced collector
    collector = EnhancedDataCollector()
    
    # Collect all enhanced data
    enhanced_data = collector.collect_all_enhanced_data(
        start_date='2020-01-01',  # Shorter period for demo
        end_date='2024-01-01'
    )
    
    # Save the data
    collector.save_enhanced_data(enhanced_data)
    
    # Print summary
    print("\n📊 Enhanced Data Collection Summary:")
    print("=" * 50)
    
    for category, category_data in enhanced_data.items():
        print(f"\n{category.upper()}:")
        for name, df in category_data.items():
            if not df.empty:
                date_range = f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}" if 'date' in df.columns else "N/A"
                print(f"  {name}: {len(df)} records, {df.shape[1]} features ({date_range})")
    
    logger.info("🎉 Enhanced data collection demo completed successfully!")


if __name__ == "__main__":
    main() 