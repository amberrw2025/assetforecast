"""
Economic Data Provider for Enhanced Forecasting
Fetches real-time economic indicators and market data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import requests
import os

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

class EconomicDataProvider:
    """Provides economic indicators and market data for enhanced forecasting."""
    
    def __init__(self):
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.fred = None
        
        if FRED_AVAILABLE and self.fred_api_key:
            try:
                self.fred = Fred(api_key=self.fred_api_key)
                print("✅ FRED API initialized")
            except Exception as e:
                print(f"⚠️ FRED API key invalid, using fallback data. Error: {e}")
        else:
            print("⚠️ FRED API not available, using simulated data")
    
    def get_economic_indicators(self, start_date: str = None, end_date: str = None) -> Dict[str, float]:
        """Get current economic indicators."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        indicators = {}
        
        # Get Federal Funds Rate
        indicators['fed_funds_rate'] = self._get_fed_funds_rate()
        
        # Get Unemployment Rate
        indicators['unemployment_rate'] = self._get_unemployment_rate()
        
        # Get 10-Year Treasury Rate
        indicators['treasury_10y'] = self._get_treasury_rate()
        
        # Get VIX (Market Volatility)
        indicators['vix'] = self._get_vix()
        
        # Get DXY (Dollar Index)
        indicators['dxy'] = self._get_dollar_index()
        
        # Get Oil Price
        indicators['oil_price'] = self._get_oil_price()
        
        # Get Gold Price
        indicators['gold_price'] = self._get_gold_price()
        
        # Calculate Economic Sentiment Score
        indicators['economic_sentiment'] = self._calculate_economic_sentiment(indicators)
        
        return indicators
    
    def _get_fed_funds_rate(self) -> float:
        """Get Federal Funds Effective Rate."""
        try:
            if self.fred:
                rate = self.fred.get_series('FEDFUNDS', limit=1).iloc[-1]
                return float(rate) if not pd.isna(rate) else 5.25
            else:
                return 5.25
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch FEDFUNDS, using fallback. Error: {e}")
            return 5.25
    
    def _get_unemployment_rate(self) -> float:
        """Get Unemployment Rate."""
        try:
            if self.fred:
                rate = self.fred.get_series('UNRATE', limit=1).iloc[-1]
                return float(rate) if not pd.isna(rate) else 3.8
            else:
                return 3.8
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch UNRATE, using fallback. Error: {e}")
            return 3.8
    
    def _get_treasury_rate(self) -> float:
        """Get 10-Year Treasury Rate."""
        try:
            if self.fred:
                rate = self.fred.get_series('GS10', limit=1).iloc[-1]
                return float(rate) if not pd.isna(rate) else 4.5
            else:
                return 4.5
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch GS10, using fallback. Error: {e}")
            return 4.5
    
    def _get_vix(self) -> float:
        """Get VIX (Volatility Index)."""
        try:
            vix_data = yf.download('^VIX', period='5d', progress=False)
            if not vix_data.empty:
                return float(vix_data['Close'].iloc[-1])
            else:
                return 20.0
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch VIX, using fallback. Error: {e}")
            return 20.0
    
    def _get_dollar_index(self) -> float:
        """Get Dollar Index (DXY)."""
        try:
            dxy_data = yf.download('DX-Y.NYB', period='5d', progress=False)
            if not dxy_data.empty:
                return float(dxy_data['Close'].iloc[-1])
            else:
                return 104.0
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch DXY, using fallback. Error: {e}")
            return 104.0
    
    def _get_oil_price(self) -> float:
        """Get Crude Oil Price."""
        try:
            oil_data = yf.download('CL=F', period='5d', progress=False)
            if not oil_data.empty:
                return float(oil_data['Close'].iloc[-1])
            else:
                return 80.0
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch Oil Price, using fallback. Error: {e}")
            return 80.0
    
    def _get_gold_price(self) -> float:
        """Get Gold Price."""
        try:
            gold_data = yf.download('GC=F', period='5d', progress=False)
            if not gold_data.empty:
                return float(gold_data['Close'].iloc[-1])
            else:
                return 2000.0
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"⚠️ Could not fetch Gold Price, using fallback. Error: {e}")
            return 2000.0
    
    def _calculate_economic_sentiment(self, indicators: Dict[str, float]) -> float:
        """Calculate overall economic sentiment score (-1 to 1)."""
        sentiment = 0.0
        
        # Federal Funds Rate impact (higher rates = negative)
        fed_rate = indicators.get('fed_funds_rate', 5.25)
        if fed_rate > 5.0:
            sentiment -= 0.2 * (fed_rate - 5.0) / 2.0
        elif fed_rate < 2.0:
            sentiment += 0.2 * (2.0 - fed_rate) / 2.0
        
        # Unemployment impact (higher unemployment = negative)
        unemployment = indicators.get('unemployment_rate', 3.8)
        if unemployment > 5.0:
            sentiment -= 0.3 * (unemployment - 5.0) / 5.0
        elif unemployment < 4.0:
            sentiment += 0.2 * (4.0 - unemployment) / 4.0
        
        # VIX impact (higher volatility = negative)
        vix = indicators.get('vix', 20.0)
        if vix > 25:
            sentiment -= 0.3 * (vix - 25) / 25
        elif vix < 15:
            sentiment += 0.2 * (15 - vix) / 15
        
        # Oil price impact (moderate prices positive)
        oil = indicators.get('oil_price', 80.0)
        if 70 <= oil <= 90:
            sentiment += 0.1
        elif oil > 120:
            sentiment -= 0.2
        
        # Clamp sentiment between -1 and 1
        return max(-1.0, min(1.0, sentiment))
    
    def get_technical_indicators(self, price_data: pd.Series, periods: int = 14) -> Dict[str, float]:
        """Calculate technical indicators from price data."""
        indicators = {}
        
        try:
            # RSI (Relative Strength Index)
            delta = price_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # Moving Average Convergence Divergence (MACD)
            ema12 = price_data.ewm(span=12).mean()
            ema26 = price_data.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
            indicators['macd_signal'] = float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0.0
            
            # Bollinger Bands
            sma20 = price_data.rolling(window=20).mean()
            std20 = price_data.rolling(window=20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            current_price = price_data.iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            indicators['bb_position'] = bb_position if not pd.isna(bb_position) else 0.5
            
            # Volatility (20-day)
            returns = price_data.pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            indicators['volatility'] = float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.2
            
            # Moving Averages
            sma_20 = price_data.rolling(window=20).mean().iloc[-1]
            sma_50 = price_data.rolling(window=50).mean().iloc[-1]
            indicators['sma_20'] = float(sma_20) if not pd.isna(sma_20) else float(price_data.iloc[-1])
            indicators['sma_50'] = float(sma_50) if not pd.isna(sma_50) else float(price_data.iloc[-1])
            indicators['price_to_sma20'] = current_price / indicators['sma_20']
            indicators['price_to_sma50'] = current_price / indicators['sma_50']
            
        except Exception as e:
            print(f"⚠️ Error calculating technical indicators: {e}")
            # Fallback values
            current_price = float(price_data.iloc[-1]) if len(price_data) > 0 else 100.0
            indicators = {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'bb_position': 0.5,
                'volatility': 0.2,
                'sma_20': current_price,
                'sma_50': current_price,
                'price_to_sma20': 1.0,
                'price_to_sma50': 1.0
            }
        
        return indicators
    
    def get_market_regime(self, indicators: Dict[str, float]) -> str:
        """Determine current market regime based on indicators."""
        economic_sentiment = indicators.get('economic_sentiment', 0.0)
        vix = indicators.get('vix', 20.0)
        rsi = indicators.get('rsi', 50.0)
        
        # High volatility regime
        if vix > 30:
            return 'volatile'
        
        # Bull market conditions
        if economic_sentiment > 0.2 and rsi < 70 and vix < 20:
            return 'bull'
        
        # Bear market conditions
        if economic_sentiment < -0.2 or vix > 25:
            return 'bear'
        
        # Default to neutral
        return 'neutral'
    
    def get_forecast_adjustments(self, 
                               base_forecast: List[float], 
                               economic_indicators: Dict[str, float],
                               technical_indicators: Dict[str, float]) -> Tuple[List[float], Dict[str, Any]]:
        """Apply economic and technical adjustments to base forecast."""
        economic_sentiment = economic_indicators.get('economic_sentiment', 0.0)
        market_regime = self.get_market_regime({**economic_indicators, **technical_indicators})
        
        # Calculate adjustment factor
        regime_multipliers = {
            'bull': 1.02,      # 2% positive bias
            'bear': 0.98,      # 2% negative bias
            'volatile': 0.99,  # 1% negative bias
            'neutral': 1.00    # No bias
        }
        
        base_multiplier = regime_multipliers.get(market_regime, 1.00)
        sentiment_adjustment = economic_sentiment * 0.01  # Max 1% adjustment
        
        adjusted_forecast = []
        for i, price in enumerate(base_forecast):
            # Diminishing effect over time
            time_decay = 0.95 ** i
            total_adjustment = ((base_multiplier - 1.0) + sentiment_adjustment) * time_decay
            adjusted_price = price * (1.0 + total_adjustment)
            adjusted_forecast.append(adjusted_price)
        
        adjustment_details = {
            'market_regime': market_regime,
            'economic_sentiment': economic_sentiment,
            'base_multiplier': base_multiplier,
            'sentiment_adjustment': sentiment_adjustment,
            'total_effect': f"{((base_multiplier - 1.0) + sentiment_adjustment) * 100:.1f}%"
        }
        
        return adjusted_forecast, adjustment_details
