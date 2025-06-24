"""
Enhanced forecasting model that integrates external data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import yfinance as yf
from datetime import datetime, timedelta

# Optional imports for enhanced functionality
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class EnhancedForecastModel:
    """Enhanced forecasting with external data integration."""
    
    def __init__(self):
        self.external_weights = {
            'economic': 0.3,
            'sentiment': 0.2,
            'technical': 0.4,
            'fundamental': 0.1
        }
        
    def get_economic_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch economic indicators that affect stock prices.
        
        Args:
            start_date (str): Start date for data
            end_date (str): End date for data
            
        Returns:
            pd.DataFrame: Economic indicators data
        """
        try:
            # Generate sample economic data for now
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            
            economic_data = {
                'fed_rate': np.random.uniform(0.5, 5.0, len(dates)),
                'unemployment': np.random.uniform(3.0, 8.0, len(dates)),
                'treasury_10y': np.random.uniform(1.0, 6.0, len(dates)),
                'vix': np.random.uniform(12, 35, len(dates))
            }
            
            df = pd.DataFrame(economic_data, index=dates)
            return df
        except Exception as e:
            print(f"Error fetching economic data: {e}")
            return pd.DataFrame()
    
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
    
    def generate_enhanced_forecast(self, ticker: str, hist_data: pd.DataFrame, 
                                 base_forecasts: Dict[str, List[float]], 
                                 days: int) -> Dict[str, Any]:
        """
        Generate enhanced forecast using external data sources.
        
        Args:
            ticker (str): Stock ticker
            hist_data (pd.DataFrame): Historical price data
            base_forecasts (Dict[str, List[float]]): Base model forecasts
            days (int): Number of days to forecast
            
        Returns:
            Dict[str, Any]: Enhanced forecast with external factors
        """
        try:
            # Get external data
            start_date = (hist_data.index[0] - timedelta(days=90)).strftime('%Y-%m-%d')
            end_date = hist_data.index[-1].strftime('%Y-%m-%d')
            
            economic_data = self.get_economic_indicators(start_date, end_date)
            sentiment_score = self.get_sentiment_score(ticker)
            fundamental_metrics = self.get_fundamental_metrics(ticker)
            technical_indicators = self.calculate_technical_indicators(hist_data['Close'])
            
            # Calculate adjustments
            adjustments = self._calculate_adjustments(
                economic_data, sentiment_score, fundamental_metrics, technical_indicators
            )
            
            # Apply adjustments to forecasts
            enhanced_forecasts = {}
            for model_name, forecast in base_forecasts.items():
                if model_name != 'Ensemble':
                    adjusted = self._apply_adjustments(forecast, adjustments)
                    enhanced_forecasts[f'{model_name}_Enhanced'] = adjusted
            
            # Create enhanced ensemble
            if enhanced_forecasts:
                ensemble = self._create_ensemble(enhanced_forecasts)
                enhanced_forecasts['Enhanced_Ensemble'] = ensemble
            
            return {
                'forecasts': enhanced_forecasts,
                'external_summary': {
                    'economic_impact': adjustments.get('economic_factor', 0.0),
                    'sentiment_impact': adjustments.get('sentiment_factor', 0.0),
                    'overall_bias': adjustments.get('overall_bias', 0.0)
                }
            }
            
        except Exception as e:
            print(f"Error in enhanced forecast: {e}")
            return {'forecasts': {}, 'external_summary': {}}
    
    def _calculate_adjustments(self, economic_data, sentiment_score, fundamentals, technical):
        """
        Calculate adjustment factors based on external data.
        
        Args:
            economic_data (pd.DataFrame): Economic indicators
            sentiment_score (float): Sentiment score
            fundamental_metrics (Dict[str, float]): Fundamental metrics
            technical_indicators (Dict[str, float]): Technical indicators
            
        Returns:
            Dict[str, float]: Adjustment factors
        """
        try:
            # Economic factor
            if not economic_data.empty:
                latest = economic_data.tail(1).iloc[0]
                economic_factor = -0.02 * (latest['fed_rate'] - 2.0)
                economic_factor = np.clip(economic_factor, -0.2, 0.2)
            else:
                economic_factor = 0.0
            
            # Sentiment factor
            sentiment_factor = sentiment_score * 0.1
            
            # Technical factor
            rsi = technical.get('rsi', 50.0)
            if rsi > 70:
                technical_factor = -0.05
            elif rsi < 30:
                technical_factor = 0.05
            else:
                technical_factor = 0.0
            
            overall_bias = (economic_factor * 0.4 + sentiment_factor * 0.3 + technical_factor * 0.3)
            
            return {
                'economic_factor': economic_factor,
                'sentiment_factor': sentiment_factor,
                'technical_factor': technical_factor,
                'overall_bias': overall_bias
            }
        except:
            return {'economic_factor': 0.0, 'sentiment_factor': 0.0, 'technical_factor': 0.0, 'overall_bias': 0.0}
    
    def _apply_adjustments(self, forecast, adjustments):
        """
        Apply external adjustments to a forecast.
        
        Args:
            forecast (List[float]): Original forecast
            adjustments (Dict[str, float]): Adjustment factors
            
        Returns:
            List[float]: Adjusted forecast
        """
        try:
            bias = adjustments.get('overall_bias', 0.0)
            adjusted = []
            for i, price in enumerate(forecast):
                bias_effect = bias * (0.9 ** i)  # Diminishing effect
                adjusted_price = price * (1 + bias_effect)
                adjusted.append(max(adjusted_price, 0.01))
            return adjusted
        except:
            return forecast
    
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