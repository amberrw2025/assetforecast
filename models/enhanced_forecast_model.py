"""
Enhanced forecasting model that integrates external data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
import sys

# Add economic data provider
sys.path.append(str(Path(__file__).parent.parent))
try:
    from economic_data_provider import EconomicDataProvider
    ECONOMIC_DATA_AVAILABLE = True
except ImportError:
    ECONOMIC_DATA_AVAILABLE = False
    print("⚠️ Economic data provider not available")

# Optional imports for enhanced functionality
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
# Load only the specific .env file
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)

class EnhancedForecastModel:
    """Enhanced forecasting with external data integration."""
    
    def __init__(self):
        self.external_weights = {
            'economic': 0.4, # Increased weight for VIX
            'sentiment': 0.1,
            'technical': 0.4,
            'fundamental': 0.1
        }
        
        # Initialize economic data provider
        if ECONOMIC_DATA_AVAILABLE:
            self.economic_provider = EconomicDataProvider()
        else:
            self.economic_provider = None
        
    def get_economic_indicators(self, start_date: str, end_date: str, ticker: str = "") -> pd.DataFrame:
        """
        Fetches economic indicators like the VIX or VFTSE Index.
        Dynamically selects the index based on the ticker.
        
        Args:
            start_date (str): Start date for data
            end_date (str): End date for data
            ticker (str): Stock ticker
            
        Returns:
            pd.DataFrame: Economic indicators data
        """
        # Determine which volatility index to use
        is_uk_stock = ticker.endswith('.L')
        volatility_ticker = '^VFTSE' if is_uk_stock else '^VIX'
        indicator_name = 'VFTSE' if is_uk_stock else 'VIX'
        
        print(f"   Fetching {indicator_name} data for {ticker}...")

        try:
            # Fetch the data from yfinance
            data = yf.download(volatility_ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No {indicator_name} data returned from yfinance.")

            # We only need the 'Close' price, let's rename it to be clear
            data = data[['Close']].rename(columns={'Close': indicator_name})
            return data
        except Exception as e:
            print(f"   ⚠️ Could not download {indicator_name} data: {e}")
            print(f"   Falling back to randomly generated data for simulation.")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            fallback_data = np.random.uniform(10, 35, size=len(dates)) # Typical VIX range
            fallback_df = pd.DataFrame(fallback_data, index=dates, columns=[indicator_name])
            return fallback_df
    
    def get_sentiment_score(self, ticker: str) -> float:
        """
        Calculate sentiment score for a given ticker.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        try:
            # Simple sentiment simulation
            np.random.seed(hash(ticker) % 1000)
            return np.random.uniform(-0.5, 0.5)
        except:
            return 0.0
    
    def get_fundamental_metrics(self, ticker: str) -> Dict[str, float]:
        """
        Get fundamental analysis metrics for a stock.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Dict[str, float]: Fundamental metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'pe_ratio': info.get('trailingPE', 15.0) or 15.0,
                'pb_ratio': info.get('priceToBook', 2.0) or 2.0,
                'roe': info.get('returnOnEquity', 0.15) or 0.15,
                'beta': info.get('beta', 1.0) or 1.0
            }
        except:
            return {'pe_ratio': 15.0, 'pb_ratio': 2.0, 'roe': 0.15, 'beta': 1.0}
    
    def calculate_technical_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate technical indicators from price data.
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            Dict[str, float]: Technical indicators
        """
        try:
            # RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return {
                'rsi': rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0,
                'volatility': prices.pct_change().std() * np.sqrt(252)
            }
        except:
            return {'rsi': 50.0, 'volatility': 0.2}
    
    def generate_enhanced_forecast(self, 
                                 ticker: str,
                                 hist_data: pd.DataFrame,
                                 forecast_dates: pd.DatetimeIndex,
                                 vix_forecast_period_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates a more realistic stock price path using a volatility index
        (VIX or VFTSE) as a guide for daily price movements.
        
        Args:
            ticker (str): Stock ticker (for potential future use).
            hist_data (pd.DataFrame): Historical price data to get the starting price.
            forecast_dates (pd.DatetimeIndex): The dates for the forecast period.
            vix_forecast_period_data (pd.DataFrame): VIX data for the forecast period.
            
        Returns:
            Dict[str, Any]: A dictionary containing the simulated forecast path.
        """
        try:
            # Get the last closing price from history as our starting point
            last_price = hist_data['Close'].iloc[-1]
            
            # Ensure last_price is a float
            if isinstance(last_price, (pd.Series, pd.DataFrame)):
                last_price = last_price.iloc[0]

            simulated_prices = []
            
            # Determine which column to use from the economic data
            indicator_name = 'VFTSE' if ticker.endswith('.L') else 'VIX'
            
            if indicator_name not in vix_forecast_period_data.columns:
                print(f"   ⚠️ {indicator_name} data not available for simulation period. Cannot generate simulated path.")
                return {'forecasts': {'VIX_Simulated_Path': None}}

            # Align VIX data with forecast dates
            aligned_vix = vix_forecast_period_data.reindex(forecast_dates, method='ffill').fillna(method='bfill')

            # Extract VIX values
            vix_values = aligned_vix.iloc[:, 0].values

            for date in forecast_dates:
                if date in aligned_vix.index:
                    vix_value = aligned_vix.loc[date, indicator_name]
                    
                    # Ensure vix_value is a float
                    if isinstance(vix_value, (pd.Series, pd.DataFrame)):
                        vix_value = vix_value.iloc[0]

                    # Simple model: volatility translates to price movement
                    # Scale it down significantly to represent daily change
                    daily_change_percentage = (vix_value / 1000) * (np.random.randn() * 0.5 + np.sign(np.random.randn()))
                    
                    last_price = last_price * (1 + daily_change_percentage)
                    simulated_prices.append(last_price)
                else:
                    # If no VIX data for a specific day, keep the price constant
                    simulated_prices.append(last_price)

            return {
                'forecasts': {'VIX_Simulated_Path': simulated_prices},
                'external_summary': {}
            }
            
        except Exception as e:
            print(f"Error in VIX-simulated forecast generation: {e}")
            return {'forecasts': {}, 'external_summary': {}}
    
    def _create_ensemble(self, forecasts):
        """
        Create enhanced ensemble forecast.
        
        Args:
            forecasts (Dict[str, List[float]]): Enhanced model forecasts
            
        Returns:
            List[float]: Enhanced ensemble forecast
        """
        try:
            if not forecasts:
                return []
            forecast_matrix = np.array(list(forecasts.values()))
            return np.mean(forecast_matrix, axis=0).tolist()
        except:
            return [] 