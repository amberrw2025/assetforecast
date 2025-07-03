"""
Enhanced Forecasting System with Market Regime Awareness and Advanced Features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import lightgbm as lgb

# Time Series Models
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Technical Analysis
import talib
from scipy import stats
from scipy.signal import argrelextrema

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AdvancedFeatureEngineer:
    """Advanced feature engineering for stock prediction"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_importance = {}
        
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close_price'].pct_change()
        df['log_returns'] = np.log(df['close_price'] / df['close_price'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'ma_{period}'] = df['close_price'].rolling(period).mean()
            df[f'ma_ratio_{period}'] = df['close_price'] / df[f'ma_{period}']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close_price'].values, timeperiod=20
        )
        df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI and momentum
        df['rsi'] = talib.RSI(df['close_price'].values, timeperiod=14)
        df['momentum'] = df['close_price'] / df['close_price'].shift(10) - 1
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close_price'].values)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_volume'] = df['close_price'] * df['volume']
            df['vwap'] = df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close_price']
        df['daily_range'] = (df['high'] - df['low']) / df['close_price']
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
        
        return df
    
    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime indicators"""
        df = df.copy()
        
        # Volatility regime
        df['vol_regime'] = pd.cut(df['volatility'], bins=3, labels=['low', 'medium', 'high'])
        df['vol_regime_num'] = pd.cut(df['volatility'], bins=3, labels=[0, 1, 2]).astype(float)
        
        # Trend regime
        df['trend_short'] = np.where(df['ma_5'] > df['ma_20'], 1, 0)
        df['trend_long'] = np.where(df['ma_20'] > df['ma_50'], 1, 0)
        df['trend_regime'] = df['trend_short'] + df['trend_long']  # 0=bearish, 1=neutral, 2=bullish
        
        # Market phase (bull/bear/sideways)
        df['market_phase'] = 0  # Default sideways
        df.loc[df['ma_ratio_50'] > 1.05, 'market_phase'] = 1  # Bull
        df.loc[df['ma_ratio_50'] < 0.95, 'market_phase'] = -1  # Bear
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Calendar features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time since features
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        return df

class MarketRegimeDetector:
    """Detect market regimes using multiple indicators"""
    
    def __init__(self):
        self.regimes = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol']
        self.current_regime = 'sideways'
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        recent_data = df.tail(30)  # Last 30 days
        
        # Calculate regime indicators
        trend_strength = recent_data['ma_ratio_20'].mean()
        volatility_level = recent_data['volatility'].mean()
        momentum = recent_data['momentum'].mean()
        
        # Define regime thresholds
        if trend_strength > 1.1 and momentum > 0.05:
            regime = 'bull'
        elif trend_strength < 0.9 and momentum < -0.05:
            regime = 'bear'
        elif volatility_level > recent_data['volatility'].quantile(0.8):
            regime = 'high_vol'
        elif volatility_level < recent_data['volatility'].quantile(0.2):
            regime = 'low_vol'
        else:
            regime = 'sideways'
        
        self.current_regime = regime
        return regime
    
    def get_regime_adjustment(self, regime: str) -> float:
        """Get adjustment factor for different regimes"""
        adjustments = {
            'bull': 1.05,
            'bear': 0.95,
            'sideways': 1.0,
            'high_vol': 0.98,
            'low_vol': 1.02
        }
        return adjustments.get(regime, 1.0)

class EnhancedMLModel:
    """Enhanced ML model with multiple algorithms"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()
        self.feature_cols = None
        self.is_fitted = False
        
    def _create_models(self):
        """Create different ML models"""
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'close_price') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML models"""
        feature_df = df.copy()
        
        # Select numeric columns and drop target
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col and 'date' not in col.lower()]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if feature_df[col].isna().sum() / len(feature_df) < 0.3]
        
        X = feature_df[feature_cols].fillna(method='ffill').fillna(0)
        y = feature_df[target_col].fillna(method='ffill')
        
        # Store feature columns for later use
        if self.feature_cols is None:
            self.feature_cols = feature_cols
        
        return X.values, y.values
    
    def fit(self, df: pd.DataFrame, target_col: str = 'close_price'):
        """Fit the enhanced ML model"""
        X, y = self.prepare_features(df, target_col)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit models
        self._create_models()
        
        # Fit each model and calculate weights based on validation performance
        val_scores = {}
        
        for name, model in self.models.items():
            try:
                # Simple train-validation split
                split_idx = int(len(X_scaled) * 0.8)
                X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate validation score
                mse = mean_squared_error(y_val, y_pred)
                val_scores[name] = 1 / (1 + mse)  # Inverse MSE as weight
                
            except Exception as e:
                print(f"Warning: Model {name} failed to fit: {e}")
                val_scores[name] = 0.01  # Small weight for failed models
        
        # Normalize weights
        total_score = sum(val_scores.values())
        self.weights = {name: score / total_score for name, score in val_scores.items()}
        
        # Refit on full data
        for model in self.models.values():
            try:
                model.fit(X_scaled, y)
            except:
                pass
        
        self.is_fitted = True
        print(f"Model weights: {self.weights}")
    
    def predict(self, df: pd.DataFrame, target_col: str = 'close_price') -> np.ndarray:
        """Make predictions using ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _ = self.prepare_features(df, target_col)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions[name] = pred
            except:
                predictions[name] = np.zeros(len(X_scaled))
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X_scaled))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred

class EnhancedForecastingSystem:
    """Complete enhanced forecasting system"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.ml_model = EnhancedMLModel()
        self.prophet_model = None
        self.is_fitted = False
        
    def prepare_stock_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare comprehensive stock data with all features"""
        stock_data = df[df['ticker'] == ticker].copy()
        stock_data = stock_data.sort_values('date').reset_index(drop=True)
        
        # Create all features
        stock_data = self.feature_engineer.create_technical_features(stock_data)
        stock_data = self.feature_engineer.create_market_regime_features(stock_data)
        stock_data = self.feature_engineer.create_time_features(stock_data)
        
        return stock_data
    
    def fit(self, df: pd.DataFrame, ticker: str):
        """Fit the enhanced forecasting system"""
        print(f"Fitting enhanced forecasting system for {ticker}...")
        
        # Prepare data
        stock_data = self.prepare_stock_data(df, ticker)
        
        # Split into train/validation (use data before 2024 for training)
        train_data = stock_data[stock_data['date'] < '2024-01-01'].copy()
        
        if len(train_data) < 100:
            raise ValueError(f"Insufficient training data for {ticker}")
        
        # Fit ML model
        self.ml_model.fit(train_data)
        
        # Fit Prophet model for comparison
        prophet_df = train_data[['date', 'close_price']].rename(columns={'date': 'ds', 'close_price': 'y'})
        prophet_df = prophet_df.dropna()
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        try:
            self.prophet_model.fit(prophet_df)
        except Exception as e:
            print(f"Prophet fitting failed: {e}")
            self.prophet_model = None
        
        self.is_fitted = True
        print(f"Enhanced forecasting system fitted for {ticker}")
    
    def predict(self, df: pd.DataFrame, ticker: str, steps: int = 30) -> Tuple[np.ndarray, str]:
        """Make enhanced predictions"""
        if not self.is_fitted:
            raise ValueError("System must be fitted before prediction")
        
        # Prepare data
        stock_data = self.prepare_stock_data(df, ticker)
        
        # Detect current market regime
        regime = self.regime_detector.detect_regime(stock_data)
        
        # Get ML predictions
        try:
            # Use recent data to predict next steps
            recent_data = stock_data.tail(100)  # Last 100 days for context
            ml_predictions = []
            
            # Iterative prediction for multiple steps
            current_data = recent_data.copy()
            
            for step in range(steps):
                # Predict next value
                pred = self.ml_model.predict(current_data.tail(1))
                ml_predictions.append(pred[0])
                
                # Update data for next prediction (simplified)
                # In practice, you'd want to update all derived features
                next_row = current_data.iloc[-1:].copy()
                next_row['close_price'] = pred[0]
                next_row['date'] = pd.to_datetime(next_row['date'].iloc[0]) + timedelta(days=1)
                
                current_data = pd.concat([current_data, next_row], ignore_index=True)
            
            ml_predictions = np.array(ml_predictions)
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            ml_predictions = np.full(steps, stock_data['close_price'].iloc[-1])
        
        # Get Prophet predictions for comparison
        prophet_predictions = None
        if self.prophet_model:
            try:
                future = self.prophet_model.make_future_dataframe(periods=steps)
                forecast = self.prophet_model.predict(future)
                prophet_predictions = forecast['yhat'].tail(steps).values
            except Exception as e:
                print(f"Prophet prediction failed: {e}")
        
        # Combine predictions with regime adjustment
        if prophet_predictions is not None:
            # Weighted combination
            combined_predictions = 0.7 * ml_predictions + 0.3 * prophet_predictions
        else:
            combined_predictions = ml_predictions
        
        # Apply regime adjustment
        regime_adjustment = self.regime_detector.get_regime_adjustment(regime)
        adjusted_predictions = combined_predictions * regime_adjustment
        
        return adjusted_predictions, regime

# Example usage function
def create_enhanced_forecast_plots(data_file: str, output_dir: str):
    """Create enhanced forecast plots with improved accuracy"""
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Define stocks to forecast
    stocks = ['NVDA', 'TSLA', 'AAPL', 'AZN.L', 'BT-A.L']  # Example stocks
    
    forecasting_system = EnhancedForecastingSystem()
    
    for ticker in stocks:
        try:
            print(f"\n=== Processing {ticker} ===")
            
            # Fit the system
            forecasting_system.fit(df, ticker)
            
            # Make predictions
            predictions, regime = forecasting_system.predict(df, ticker, steps=30)
            
            print(f"Forecast for {ticker} completed. Market regime: {regime}")
            print(f"Predicted price range: ${predictions.min():.2f} - ${predictions.max():.2f}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

if __name__ == "__main__":
    # Example usage
    create_enhanced_forecast_plots(
        "data/combined_ftse_sp500_data.csv",
        "enhanced_forecast_plots"
    ) 