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

from config import ECONOMIC_INDICATORS, DATA_SOURCES, FRED_API_KEY
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
                logger.warning("FRED API key not provided. Using sample data.")
                return self._generate_sample_fred_data(series_id, description)
            
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
    
    def _generate_sample_fred_data(self, series_id: str, description: str = None) -> pd.DataFrame:
        """
        Generate sample FRED data for testing purposes.
        
        Args:
            series_id (str): Series ID
            description (str): Description
            
        Returns:
            pd.DataFrame: Sample economic data
        """
        # Generate date range
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Generate sample data based on series type
        if "FEDFUNDS" in series_id:
            # Federal Funds Rate - typically between 0-10%
            values = np.random.uniform(0.5, 5.5, len(dates))
        elif "UNRATE" in series_id:
            # Unemployment Rate - typically between 3-15%
            values = np.random.uniform(3.0, 8.0, len(dates))
        elif "IR3TIB" in series_id:
            # UK Base Rate - typically between 0-15%
            values = np.random.uniform(0.1, 5.0, len(dates))
        else:
            # Generic economic indicator
            values = np.random.uniform(50, 150, len(dates))
        
        # Add some trend and seasonality
        trend = np.linspace(0, 1, len(dates))
        seasonality = np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        values = values + trend * 0.5 + seasonality * 0.2
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'series_id': series_id,
            'description': description or series_id,
            'source': 'FRED (Sample)'
        })
        
        logger.info(f"Generated sample FRED data for {series_id}")
        return df
    
    def get_eia_data(self, series_id: str, description: str = None) -> pd.DataFrame:
        """
        Fetch data from EIA (Energy Information Administration).
        
        Args:
            series_id (str): EIA series ID
            description (str): Description of the series
            
        Returns:
            pd.DataFrame: Energy data series
        """
        try:
            # EIA API endpoint
            base_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
            params = {
                "api_key": "DEMO_KEY",  # Replace with actual EIA API key
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
            
            # If API fails, generate sample data
            logger.warning(f"EIA API failed for {series_id}. Using sample data.")
            return self._generate_sample_eia_data(series_id, description)
            
        except Exception as e:
            logger.error(f"Error fetching EIA data for {series_id}: {str(e)}")
            return self._generate_sample_eia_data(series_id, description)
    
    def _generate_sample_eia_data(self, series_id: str, description: str = None) -> pd.DataFrame:
        """
        Generate sample EIA data for testing purposes.
        
        Args:
            series_id (str): Series ID
            description (str): Description
            
        Returns:
            pd.DataFrame: Sample energy data
        """
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Generate sample oil price data
        if "RBRTE" in series_id:  # Brent Crude
            base_price = 80.0
            volatility = 0.1
        else:
            base_price = 100.0
            volatility = 0.05
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, volatility, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 10))  # Minimum price of $10
        
        df = pd.DataFrame({
            'date': dates,
            'value': prices,
            'series_id': series_id,
            'description': description or series_id,
            'source': 'EIA (Sample)'
        })
        
        logger.info(f"Generated sample EIA data for {series_id}")
        return df
    
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
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching Google Trends data for {keyword}: {str(e)}")
                    continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"Successfully fetched Google Trends data for {len(keywords)} keywords")
                return combined_df
            else:
                return pd.DataFrame()
                
        except ImportError:
            logger.error("pytrends not installed. Please install with: pip install pytrends")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in Google Trends data collection: {str(e)}")
            return pd.DataFrame()
    
    def collect_all_economic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect all economic data from various sources.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of economic data by source
        """
        economic_data = {}
        
        # Collect FRED data
        logger.info("Collecting FRED economic data...")
        fred_data = []
        
        for indicator_name, config in ECONOMIC_INDICATORS.items():
            if config['source'] == 'FRED':
                data = self.get_fred_data(
                    config['series_id'], 
                    config['description']
                )
                if not data.empty:
                    data['indicator_name'] = indicator_name
                    fred_data.append(data)
        
        if fred_data:
            economic_data['fred'] = pd.concat(fred_data, ignore_index=True)
        
        # Collect EIA data
        logger.info("Collecting EIA energy data...")
        eia_data = []
        
        for indicator_name, config in ECONOMIC_INDICATORS.items():
            if config['source'] == 'EIA':
                data = self.get_eia_data(
                    config['series_id'],
                    config['description']
                )
                if not data.empty:
                    data['indicator_name'] = indicator_name
                    eia_data.append(data)
        
        if eia_data:
            economic_data['eia'] = pd.concat(eia_data, ignore_index=True)
        
        # Collect Google Trends data (optional)
        logger.info("Collecting Google Trends data...")
        keywords = ['inflation', 'recession', 'interest rates', 'oil prices', 'unemployment']
        trends_data = self.get_google_trends_data(keywords)
        
        if not trends_data.empty:
            economic_data['google_trends'] = trends_data
        
        logger.info(f"Economic data collection completed. Sources: {list(economic_data.keys())}")
        return economic_data
    
    def save_economic_data(self, economic_data: Dict[str, pd.DataFrame]):
        """
        Save economic data to CSV files.
        
        Args:
            economic_data (Dict[str, pd.DataFrame]): Economic data by source
        """
        from config import RAW_DATA_DIR
        
        saved_files = []
        
        for source, data in economic_data.items():
            if not data.empty:
                filename = f"economic_{source}.csv"
                filepath = RAW_DATA_DIR / filename
                data.to_csv(filepath, index=False)
                saved_files.append(filepath)
                logger.info(f"Economic data saved to {filepath}")
        
        return saved_files

def main():
    """
    Main function to run economic data collection.
    """
    collector = EconomicDataCollector()
    
    # Collect all economic data
    economic_data = collector.collect_all_economic_data()
    
    if economic_data:
        # Save data
        saved_files = collector.save_economic_data(economic_data)
        logger.info(f"Economic data collection completed. Files saved: {saved_files}")
        
        # Print summary
        print(f"\nEconomic Data Collection Summary:")
        for source, data in economic_data.items():
            print(f"{source.upper()}: {len(data)} records, {data['date'].nunique()} unique dates")
    else:
        logger.error("No economic data collected. Please check your configuration and API access.")

if __name__ == "__main__":
    main() 