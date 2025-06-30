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