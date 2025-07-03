"""
Economic data acquisition module for the Forecast Accuracy Assessment Model.
Handles collection of macroeconomic and market data from various sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import time
from tqdm import tqdm

from config import ECONOMIC_INDICATORS, DATA_SOURCES, FRED_API_KEY, EIA_API_KEY
from utils.logger import get_logger

logger = get_logger("economic_data")

class EconomicDataCollector:
    """
    Collects economic and market data from various sources including FRED, EIA, etc.
    """
    
    def __init__(self):
        self.start_date = DATA_SOURCES["start_date"]
        self.end_date = DATA_SOURCES.get("end_date") or datetime.now().strftime("%Y-%m-%d")
        self.fred_api_key = FRED_API_KEY
        self.eia_api_key = EIA_API_KEY
        
    def get_fred_data(self, series_id: str, description: str = None) -> pd.DataFrame:
        """
        Fetch data from FRED (Federal Reserve Economic Data).
        
        Args:
            series_id (str): FRED series ID
            description (str): Description of the series
            
        Returns:
            pd.DataFrame: Economic data series
        """
        try:
            if not self.fred_api_key:
                logger.warning("FRED API key not provided. Skipping FRED data.")
                return pd.DataFrame()
            
            # Import fredapi only if API key is available
            try:
                from fredapi import Fred
                fred = Fred(api_key=self.fred_api_key)
                
                # Fetch data
                data = fred.get_series(series_id, start=self.start_date, end=self.end_date)
                
                df = pd.DataFrame({
                    'date': data.index,
                    'value': data.values,
                    'series_id': series_id,
                    'description': description or series_id,
                    'source': 'FRED'
                })
                
                logger.info(f"Successfully fetched FRED data for {series_id}")
                return df
                
            except ImportError:
                logger.error("fredapi not installed. Please install with: pip install fredapi")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_eia_data(self, series_id: str, description: str = None) -> pd.DataFrame:
        """
        Fetch data from EIA (Energy Information Administration).
        
        Args:
            series_id (str): EIA series ID
            description (str): Description of the series
            
        Returns:
            pd.DataFrame: Energy data series
        """
        if not self.eia_api_key or self.eia_api_key == "DEMO_KEY":
            logger.warning("EIA API key not provided. Skipping EIA data.")
            return pd.DataFrame()

        try:
            # EIA API endpoint
            base_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
            params = {
                "api_key": self.eia_api_key,
                "series_id": series_id,
                "start": self.start_date,
                "end": self.end_date,
                "frequency": "daily"
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse EIA response
                if 'response' in data and 'data' in data['response']:
                    records = data['response']['data']
                    
                    df = pd.DataFrame(records)
                    df['date'] = pd.to_datetime(df['period'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df['series_id'] = series_id
                    df['description'] = description or series_id
                    df['source'] = 'EIA'
                    
                    # Select relevant columns
                    df = df[['date', 'value', 'series_id', 'description', 'source']]
                    
                    logger.info(f"Successfully fetched EIA data for {series_id}")
                    return df
            
            logger.warning(f"EIA API failed for {series_id}. Skipping.")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching EIA data for {series_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_google_trends_data(self, keywords: List[str], geo: str = "US") -> pd.DataFrame:
        """
        Fetch Google Trends data for search interest.
        
        Args:
            keywords (List[str]): Keywords to search for
            geo (str): Geographic location
            
        Returns:
            pd.DataFrame: Google Trends data
        """
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            
            all_data = []
            
            for keyword in tqdm(keywords, desc="Fetching Google Trends data"):
                try:
                    # Build payload
                    pytrends.build_payload([keyword], cat=0, timeframe=f'{self.start_date} {self.end_date}', geo=geo)
                    
                    # Get interest over time
                    data = pytrends.interest_over_time()
                    
                    if not data.empty:
                        df = data.reset_index()
                        df['keyword'] = keyword
                        df['geo'] = geo
                        df['source'] = 'Google Trends'
                        
                        # Rename columns
                        df = df.rename(columns={
                            'date': 'date',
                            keyword: 'value',
                            'isPartial': 'is_partial'
                        })
                        
                        all_data.append(df)
                        
                except Exception as e:
                    logger.warning(f"Error fetching Google Trends for '{keyword}': {str(e)}")
                    # Add a small delay to avoid rate limiting issues
                    time.sleep(2)
            
            if not all_data:
                return pd.DataFrame(columns=['date', 'value', 'keyword', 'geo', 'source', 'is_partial'])
                
            return pd.concat(all_data, ignore_index=True)
            
        except ImportError:
            logger.error("pytrends not installed. Please install with: pip install pytrends")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends data: {str(e)}")
            return pd.DataFrame()

    def get_yfinance_data(self, ticker_config: Dict, data_type: str = "market_data") -> Dict[str, pd.DataFrame]:
        """
        Enhanced method to collect data from Yahoo Finance (ETFs, indices, FX, commodities).
        
        Args:
            ticker_config: Dictionary with ticker configurations
            data_type: Type of data being collected
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of dataframes
        """
        import yfinance as yf
        
        collected_data = {}
        
        for name, info in tqdm(ticker_config.items(), desc=f"Collecting {data_type}"):
            ticker = info["ticker"]
            description = info["description"]
            
            try:
                # Download data from Yahoo Finance
                data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                
                if not data.empty:
                    # Handle new yfinance multi-level columns
                    if isinstance(data.columns, pd.MultiIndex):
                        # Flatten multi-level columns
                        df = pd.DataFrame({
                            'date': data.index,
                            'open': data[[col for col in data.columns if 'Open' in col[0]][0]],
                            'high': data[[col for col in data.columns if 'High' in col[0]][0]],
                            'low': data[[col for col in data.columns if 'Low' in col[0]][0]],
                            'close': data[[col for col in data.columns if 'Close' in col[0]][0]],
                            'volume': data[[col for col in data.columns if 'Volume' in col[0]][0]],
                        }).reset_index(drop=True)
                    else:
                        df = pd.DataFrame({
                            'date': data.index,
                            'open': data['Open'],
                            'high': data['High'],
                            'low': data['Low'],
                            'close': data['Close'],
                            'volume': data['Volume'],
                        }).reset_index(drop=True)
                    
                    # Add metadata
                    df['ticker'] = ticker
                    df['name'] = name
                    df['description'] = description
                    df['source'] = 'Yahoo Finance'
                    df['data_type'] = data_type
                    
                    # Calculate additional features
                    df['return_1d'] = df['close'].pct_change()
                    df['return_5d'] = df['close'].pct_change(5)
                    df['return_20d'] = df['close'].pct_change(20)
                    df['volatility_20d'] = df['return_1d'].rolling(20).std()
                    df['sma_20'] = df['close'].rolling(20).mean()
                    df['sma_50'] = df['close'].rolling(50).mean()
                    
                    collected_data[name] = df
                    logger.info(f"Successfully collected {name} ({ticker}): {len(df)} records")
                    
                else:
                    logger.warning(f"No data received for {name} ({ticker})")
                    
            except Exception as e:
                logger.error(f"Error collecting {name} ({ticker}): {str(e)}")
                
            # Small delay to be respectful to Yahoo Finance
            time.sleep(0.1)
        
        return collected_data

    def collect_all_economic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect all defined economic indicators.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of economic dataframes
        """
        economic_data = {}
        
        for name, params in tqdm(ECONOMIC_INDICATORS.items(), desc="Collecting Economic Data"):
            source = params.get("source")
            series_id = params.get("series_id")
            description = params.get("description")
            
            if source == "FRED":
                df = self.get_fred_data(series_id, description)
            elif source == "EIA":
                df = self.get_eia_data(series_id, description)
            elif source == "Google Trends":
                keywords = params.get("keywords", [])
                geo = params.get("geo", "US")
                df = self.get_google_trends_data(keywords, geo)
            else:
                logger.warning(f"Unknown data source '{source}' for indicator '{name}'")
                continue
            
            if not df.empty:
                economic_data[name] = df
        
        return economic_data
    
    def collect_enhanced_market_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect all enhanced market data sources (sectors, volatility, FX, bonds).
        
        Returns:
            Dict containing all enhanced datasets organized by category
        """
        from config import SECTOR_ETFS, VOLATILITY_INDICES, CURRENCY_COMMODITIES, BOND_INDICATORS
        
        logger.info("🚀 Starting enhanced market data collection...")
        
        enhanced_data = {}
        
        # Collect sector ETF data
        try:
            sector_data = self.get_yfinance_data(SECTOR_ETFS, "sector_etf")
            if sector_data:
                enhanced_data['sectors'] = sector_data
                logger.info(f"✅ Sector ETFs: {len(sector_data)} collected")
        except Exception as e:
            logger.error(f"❌ Error collecting sector data: {str(e)}")
        
        # Collect volatility indices
        try:
            volatility_data = self.get_yfinance_data(VOLATILITY_INDICES, "volatility_index")
            if volatility_data:
                enhanced_data['volatility'] = volatility_data
                logger.info(f"✅ Volatility indices: {len(volatility_data)} collected")
        except Exception as e:
            logger.error(f"❌ Error collecting volatility data: {str(e)}")
        
        # Collect currency and commodity data
        try:
            fx_commodity_data = self.get_yfinance_data(CURRENCY_COMMODITIES, "fx_commodity")
            if fx_commodity_data:
                enhanced_data['fx_commodities'] = fx_commodity_data
                logger.info(f"✅ FX/Commodities: {len(fx_commodity_data)} collected")
        except Exception as e:
            logger.error(f"❌ Error collecting FX/commodity data: {str(e)}")
        
        # Collect bond indicators
        try:
            bond_data = self.get_yfinance_data(BOND_INDICATORS, "bond_indicator")
            if bond_data:
                enhanced_data['bonds'] = bond_data
                logger.info(f"✅ Bond indicators: {len(bond_data)} collected")
        except Exception as e:
            logger.error(f"❌ Error collecting bond data: {str(e)}")
        
        logger.info(f"🎉 Enhanced market data collection complete! Categories: {list(enhanced_data.keys())}")
        return enhanced_data

    def save_economic_data(self, economic_data: Dict[str, pd.DataFrame]):
        """
        Save economic data to CSV files.
        
        Args:
            economic_data (Dict[str, pd.DataFrame]): Data to save
        """
        for name, df in economic_data.items():
            path = f"data/raw/economic_{name}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Saved economic data for {name} to {path}")

def main():
    """Main function to collect and save economic data."""
    logger.info("Starting economic data collection...")
    
    collector = EconomicDataCollector()
    economic_data = collector.collect_all_economic_data()
    collector.save_economic_data(economic_data)
    
    logger.info("Economic data collection finished successfully.")

if __name__ == "__main__":
    main()