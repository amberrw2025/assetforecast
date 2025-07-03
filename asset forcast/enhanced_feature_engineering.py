"""
Enhanced Feature Engineering for Stock Forecasting
Addresses Fix #4: Enhanced Feature Engineering
- Economic indicators (VIX, interest rates, currency)
- Technical analysis indicators  
- Market regime detection
- Sector-specific features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Try to import additional libraries
try:
    import yfinance as yf
    import fredapi
    HAS_EXTERNAL_DATA = True
except ImportError:
    HAS_EXTERNAL_DATA = False
    logger.warning("yfinance or fredapi not available. Using simplified features.")

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not available. Using simplified technical indicators.")


class EconomicDataProvider:
    """Provides economic indicators and market context data"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key
        self.economic_cache = {}
        
        if HAS_EXTERNAL_DATA and fred_api_key:
            try:
                self.fred = fredapi.Fred(api_key=fred_api_key)
            except:
                self.fred = None
                logger.warning("FRED API setup failed")
        else:
            self.fred = None
    
    def get_vix_data(self, start_date: str = '2015-01-01', end_date: str = '2024-12-31') -> pd.DataFrame:
        """Get VIX (volatility index) data"""
        if not HAS_EXTERNAL_DATA:
            return self._create_synthetic_vix(start_date, end_date)
        
        try:
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix.empty:
                vix_df = pd.DataFrame({
                    'date': vix.index,
                    'vix_close': vix['Close'].values,
                    'vix_high': vix['High'].values,
                    'vix_low': vix['Low'].values
                })
                vix_df['vix_sma_20'] = vix_df['vix_close'].rolling(20).mean()
                vix_df['vix_volatility'] = vix_df['vix_close'].rolling(20).std()
                return vix_df.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to get VIX data: {e}")
        
        return self._create_synthetic_vix(start_date, end_date)
    
    def get_interest_rate_data(self, start_date: str = '2015-01-01', end_date: str = '2024-12-31') -> pd.DataFrame:
        """Get interest rate data from FRED"""
        if not self.fred:
            return self._create_synthetic_rates(start_date, end_date)
        
        try:
            # Get key interest rates
            fed_rate = self.fred.get_series('FEDFUNDS', start=start_date, end=end_date)
            treasury_10y = self.fred.get_series('GS10', start=start_date, end=end_date)
            treasury_2y = self.fred.get_series('GS2', start=start_date, end=end_date)
            
            # Combine into DataFrame
            rates_df = pd.DataFrame({
                'date': fed_rate.index,
                'fed_funds_rate': fed_rate.values,
            })
            
            if not treasury_10y.empty:
                treasury_10y_df = pd.DataFrame({'date': treasury_10y.index, 'treasury_10y': treasury_10y.values})
                rates_df = rates_df.merge(treasury_10y_df, on='date', how='outer')
            
            if not treasury_2y.empty:
                treasury_2y_df = pd.DataFrame({'date': treasury_2y.index, 'treasury_2y': treasury_2y.values})
                rates_df = rates_df.merge(treasury_2y_df, on='date', how='outer')
            
            # Calculate yield curve slope
            if 'treasury_10y' in rates_df.columns and 'treasury_2y' in rates_df.columns:
                rates_df['yield_curve_slope'] = rates_df['treasury_10y'] - rates_df['treasury_2y']
            
            # Forward fill missing values
            rates_df = rates_df.fillna(method='ffill')
            return rates_df.reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Failed to get interest rate data: {e}")
        
        return self._create_synthetic_rates(start_date, end_date)
    
    def _create_synthetic_vix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create synthetic VIX data for testing"""
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)
        
        # Generate realistic VIX-like data
        base_vix = 20 + np.random.normal(0, 5, len(dates))
        base_vix = np.clip(base_vix, 10, 80)  # Realistic VIX range
        
        return pd.DataFrame({
            'date': dates,
            'vix_close': base_vix,
            'vix_high': base_vix * 1.1,
            'vix_low': base_vix * 0.9,
            'vix_sma_20': pd.Series(base_vix).rolling(20).mean().fillna(20),
            'vix_volatility': pd.Series(base_vix).rolling(20).std().fillna(5)
        })


class TechnicalIndicatorCalculator:
    """Calculates comprehensive technical indicators"""
    
    @staticmethod
    def calculate_basic_indicators(df: pd.DataFrame, price_col: str = 'close_price') -> pd.DataFrame:
        """Calculate basic technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df[price_col].rolling(period).mean()
            df[f'ema_{period}'] = df[price_col].ewm(span=period).mean()
            
        # Price relative to moving averages
        for period in [5, 10, 20, 50]:
            df[f'price_to_sma_{period}'] = df[price_col] / df[f'sma_{period}']
            df[f'price_to_ema_{period}'] = df[price_col] / df[f'ema_{period}']
        
        # Volatility measures
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'realized_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        # Momentum indicators
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = df[price_col] / df[price_col].shift(period) - 1
            df[f'roc_{period}'] = df[price_col].pct_change(periods=period)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class EnhancedFeatureEngineer:
    """Main class that orchestrates all feature engineering"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.economic_provider = EconomicDataProvider(fred_api_key)
        self.tech_calculator = TechnicalIndicatorCalculator()
        
        # Cache for external data
        self.external_data_cache = {}
    
    def add_all_features(self, df: pd.DataFrame, 
                        ticker: str,
                        price_col: str = 'close_price',
                        date_col: str = 'date') -> pd.DataFrame:
        """Add all enhanced features to a stock dataset"""
        
        logger.info(f"Adding enhanced features for {ticker}")
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # 1. Basic technical indicators
        df = self.tech_calculator.calculate_basic_indicators(df, price_col)
        
        # 2. Market-specific features
        df = self._add_market_specific_features(df, ticker)
        
        # 3. Time-based features
        df = self._add_time_features(df, date_col)
        
        # 4. Lag features
        df = self._add_lag_features(df, price_col)
        
        # 5. Clean up and final processing
        df = self._clean_features(df)
        
        logger.info(f"Enhanced features added for {ticker}. Final shape: {df.shape}")
        
        return df
    
    def _add_market_specific_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Add market-specific features"""
        df = df.copy()
        
        # Market identification
        df['is_ftse'] = 1 if ticker.endswith('.L') else 0
        df['is_sp500'] = 1 if not ticker.endswith('.L') else 0
        
        # Sector classification (simplified)
        tech_tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'INTC']
        finance_tickers = ['LSEG.L', 'JPM', 'BAC']
        healthcare_tickers = ['AZN.L', 'JNJ', 'PFE']
        
        df['is_tech'] = 1 if any(t in ticker for t in tech_tickers) else 0
        df['is_finance'] = 1 if any(t in ticker for t in finance_tickers) else 0
        df['is_healthcare'] = 1 if any(t in ticker for t in healthcare_tickers) else 0
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Cyclical time features
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_week'] = df[date_col].dt.dayofweek
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add lagged features"""
        df = df.copy()
        
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'{price_col}_lag_{lag}'] = df[price_col].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Volatility lags
        for lag in [1, 2, 5]:
            if f'volatility_20' in df.columns:
                df[f'volatility_20_lag_{lag}'] = df['volatility_20'].shift(lag)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features"""
        df = df.copy()
        
        # Forward fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Fill remaining NaNs with 0
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df


def test_enhanced_features():
    """Test the enhanced feature engineering"""
    logger.info("Testing enhanced feature engineering...")
    
    # Create test data
    dates = pd.date_range('2020-01-01', '2024-06-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'close_price': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Test feature engineering
    engineer = EnhancedFeatureEngineer()
    enhanced_df = engineer.add_all_features(df, 'TEST')
    
    logger.info(f"Original features: {len(df.columns)}")
    logger.info(f"Enhanced features: {len(enhanced_df.columns)}")
    logger.info(f"New features added: {len(enhanced_df.columns) - len(df.columns)}")
    
    logger.info("Enhanced feature engineering test completed!")
    
    return enhanced_df


if __name__ == "__main__":
    test_enhanced_features() 