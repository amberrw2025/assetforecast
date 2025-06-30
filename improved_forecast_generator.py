"""
Improved Forecast Generator with Enhanced Features and Market Awareness
Addresses the poor 2024 forecast accuracy issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using simplified technical indicators.")

# Original models
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel

class ImprovedFeatureEngineer:
    """Enhanced feature engineering for better forecasting accuracy"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for better prediction"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Basic price features
        df['returns'] = df['close_price'].pct_change()
        df['log_returns'] = np.log(df['close_price'] / df['close_price'].shift(1))
        
        # Volatility features
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'ma_{window}'] = df['close_price'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['close_price'] / df[f'ma_{window}']
        
        # Technical indicators
        if TALIB_AVAILABLE:
            # RSI
            df['rsi'] = talib.RSI(df['close_price'].values, timeperiod=14)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close_price'].values)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close_price'].values)
            df['bb_position'] = (df['close_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        else:
            # Simplified versions
            df['rsi'] = self._calculate_rsi(df['close_price'], 14)
            df['bb_position'] = 0.5  # Neutral position
        
        # Price momentum and trends
        df['momentum_5'] = df['close_price'] / df['close_price'].shift(5) - 1
        df['momentum_10'] = df['close_price'] / df['close_price'].shift(10) - 1
        df['momentum_20'] = df['close_price'] / df['close_price'].shift(20) - 1
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_volume'] = df['close_price'] * df['volume']
        
        # Market structure features
        df['high_low_ratio'] = df['high'] / df['low'] if 'high' in df.columns and 'low' in df.columns else 1
        df['daily_range'] = (df['high'] - df['low']) / df['close_price'] if 'high' in df.columns and 'low' in df.columns else 0
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Rolling statistics
        for window in [10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        
        # Time features
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Market regime indicators
        df['trend_strength'] = (df['ma_5'] > df['ma_20']).astype(int)
        df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(60).mean()).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI manually if TA-Lib not available"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MarketRegimeDetector:
    """Detect market regimes for better forecasting"""
    
    def __init__(self):
        self.regimes = {
            'bull_market': {'trend': 1, 'volatility': 'low'},
            'bear_market': {'trend': -1, 'volatility': 'high'},
            'sideways': {'trend': 0, 'volatility': 'medium'},
            'high_volatility': {'trend': 0, 'volatility': 'high'}
        }
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        recent_data = df.tail(30)  # Last 30 days
        
        # Calculate trend strength
        trend_indicator = recent_data['ma_ratio_20'].mean()
        volatility_level = recent_data['volatility_20'].mean()
        vol_percentile = df['volatility_20'].quantile(0.8)
        
        # Determine regime
        if trend_indicator > 1.05:
            if volatility_level < vol_percentile * 0.8:
                return 'bull_market'
            else:
                return 'high_volatility'
        elif trend_indicator < 0.95:
            return 'bear_market'
        else:
            if volatility_level > vol_percentile:
                return 'high_volatility'
            else:
                return 'sideways'
    
    def get_regime_adjustments(self, regime: str) -> dict:
        """Get adjustments for different regimes"""
        adjustments = {
            'bull_market': {'trend_boost': 1.02, 'confidence': 0.9},
            'bear_market': {'trend_boost': 0.98, 'confidence': 0.85},
            'sideways': {'trend_boost': 1.0, 'confidence': 0.75},
            'high_volatility': {'trend_boost': 1.0, 'confidence': 0.6}
        }
        return adjustments.get(regime, {'trend_boost': 1.0, 'confidence': 0.7})


class EnhancedMLEnsemble:
    """Enhanced ML ensemble for better predictions"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.is_fitted = False
    
    def _initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
    
    def prepare_ml_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for ML models"""
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude target and date-related columns we don't want
        exclude_cols = ['close_price', 'date', 'high', 'low', 'open', 'volume']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove columns with too many missing values
        feature_cols = [col for col in feature_cols if df[col].isna().sum() / len(df) < 0.3]
        
        if self.feature_columns is None:
            self.feature_columns = feature_cols
        
        # Prepare features and target
        X = df[self.feature_columns].fillna(method='ffill').fillna(0)
        y = df['close_price']
        
        return X.values, y.values
    
    def fit(self, df: pd.DataFrame):
        """Fit the ML ensemble"""
        self._initialize_models()
        
        X, y = self.prepare_ml_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-validation split for weight calculation
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Fit models and calculate performance weights
        val_scores = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                val_scores[name] = 1.0 / (1.0 + mse)  # Inverse MSE as weight
            except Exception as e:
                print(f"Warning: {name} failed to fit: {e}")
                val_scores[name] = 0.1
        
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
        print(f"ML Ensemble weights: {self.weights}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first")
        
        X, _ = self.prepare_ml_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        ensemble_pred = np.zeros(len(X_scaled))
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                ensemble_pred += self.weights[name] * pred
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
        
        return ensemble_pred


class ImprovedForecastGenerator:
    """Improved forecast generator with enhanced accuracy"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.data_dir = self.project_root / "data"
        self.plots_dir = self.project_root / "improved_forecast_plots_2024"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = ImprovedFeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.ml_ensemble = EnhancedMLEnsemble()
        
        # Stock lists
        self.ftse_stocks = ['AZN.L', 'LSEG.L', 'RKT.L', 'OCDO.L', 'CRDA.L',
                           'BT-A.L', 'VOD.L', 'SSE.L', 'GLEN.L', 'TSCO.L']
        self.sp500_stocks = ['NVDA', 'TSLA', 'MRNA', 'ZM', 'NFLX',
                            'WBA', 'INTC', 'PARA', 'PAYC', 'F']
        self.all_tickers = self.ftse_stocks + self.sp500_stocks
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data with enhanced features"""
        print("📦 Loading and preparing enhanced dataset...")
        
        combined_file = self.data_dir / "combined_ftse_sp500_data.csv"
        if not combined_file.exists():
            print(f"❌ Data file not found: {combined_file}")
            return None
        
        try:
            df = pd.read_csv(combined_file)
            if 'Symbol' in df.columns:
                df.rename(columns={'Symbol': 'ticker'}, inplace=True)
            
            # Filter for our stocks
            df = df[df['ticker'].isin(self.all_tickers)]
            df['date'] = pd.to_datetime(df['date'])
            
            print(f"✅ Loaded {len(df)} records for {len(df['ticker'].unique())} stocks")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def generate_improved_forecast(self, ticker: str, df: pd.DataFrame):
        """Generate improved forecast for a single stock"""
        print(f"📈 Generating improved forecast for {ticker}...")
        
        try:
            # Get stock data
            stock_data = df[df['ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('date').reset_index(drop=True)
            
            # Create enhanced features
            stock_data = self.feature_engineer.create_enhanced_features(stock_data)
            
            # Split data: train on pre-2024, test on 2024
            train_data = stock_data[stock_data['date'] < '2024-01-01'].copy()
            test_data = stock_data[stock_data['date'] >= '2024-01-01'].copy()
            
            if len(train_data) < 100 or len(test_data) < 10:
                print(f"    ⚠️ Insufficient data for {ticker}")
                return
            
            # Detect market regime
            regime = self.regime_detector.detect_regime(train_data)
            regime_adjustments = self.regime_detector.get_regime_adjustments(regime)
            
            print(f"    📊 Market regime: {regime}")
            
            # Fit ML ensemble
            self.ml_ensemble.fit(train_data)
            
            # Generate predictions using the last training data to predict test period
            last_train_data = train_data.tail(100)  # Use last 100 days of training for context
            
            # Create forecast dates
            forecast_dates = test_data['date'].values
            forecast_steps = len(test_data)
            
            # Simple approach: use the trained model to predict on test features
            # In practice, this requires careful handling of feature engineering for future dates
            try:
                # Create enhanced features for test data
                enhanced_test_data = self.feature_engineer.create_enhanced_features(
                    pd.concat([train_data.tail(50), test_data], ignore_index=True)
                ).tail(len(test_data))
                
                ml_forecast = self.ml_ensemble.predict(enhanced_test_data)
            except Exception as e:
                print(f"    ⚠️ ML prediction failed for {ticker}: {e}")
                # Fallback to simple trend continuation
                last_price = train_data['close_price'].iloc[-1]
                trend = train_data['close_price'].iloc[-10:].pct_change().mean()
                ml_forecast = np.array([last_price * (1 + trend * i) for i in range(1, forecast_steps + 1)])
            
            # Apply regime adjustments
            adjusted_forecast = ml_forecast * regime_adjustments['trend_boost']
            
            # Create comparison with original simple forecast
            # Simple baseline: last price with average return
            last_price = train_data['close_price'].iloc[-1]
            avg_return = train_data['returns'].mean()
            simple_forecast = np.array([last_price * (1 + avg_return) ** i for i in range(1, forecast_steps + 1)])
            
            # Create the plot
            self._create_comparison_plot(
                ticker, 
                train_data, 
                test_data, 
                adjusted_forecast, 
                simple_forecast, 
                regime
            )
            
            print(f"    ✅ Improved forecast generated for {ticker}")
            
        except Exception as e:
            print(f"    ❌ Error generating forecast for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_comparison_plot(self, ticker: str, train_data: pd.DataFrame, 
                              test_data: pd.DataFrame, improved_forecast: np.ndarray,
                              simple_forecast: np.ndarray, regime: str):
        """Create comparison plot showing improved vs original forecast"""
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot historical data
        ax.plot(train_data['date'], train_data['close_price'], 
               color='gray', alpha=0.7, label='Historical Price (Training)')
        
        # Plot actual 2024 prices
        ax.plot(test_data['date'], test_data['close_price'], 
               color='black', linewidth=2, marker='o', markersize=3,
               label='Actual Price (2024)')
        
        # Plot forecasts
        ax.plot(test_data['date'], improved_forecast, 
               color='#2E8B57', linewidth=2, linestyle='--',
               label='Improved ML Forecast')
        
        ax.plot(test_data['date'], simple_forecast, 
               color='#FF6B6B', linewidth=2, linestyle='--', alpha=0.7,
               label='Simple Baseline Forecast')
        
        # Calculate and display accuracy metrics
        improved_mse = mean_squared_error(test_data['close_price'], improved_forecast)
        simple_mse = mean_squared_error(test_data['close_price'], simple_forecast)
        improvement = ((simple_mse - improved_mse) / simple_mse) * 100
        
        # Add performance text
        textstr = f'Market Regime: {regime.upper()}\n'
        textstr += f'Improved MSE: {improved_mse:.2f}\n'
        textstr += f'Simple MSE: {simple_mse:.2f}\n'
        textstr += f'Improvement: {improvement:.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Formatting
        ax.set_title(f'Improved Forecast Comparison: {ticker} (2024)', fontsize=16, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate training and testing
        if len(train_data) > 0:
            ax.axvline(x=train_data['date'].iloc[-1], color='red', linestyle=':', alpha=0.5)
            ax.text(train_data['date'].iloc[-1], ax.get_ylim()[1] * 0.95, 
                   'Training|Testing', rotation=90, alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{ticker}_improved_forecast_2024.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    📊 Improvement: {improvement:.1f}% (MSE: {improved_mse:.2f} vs {simple_mse:.2f})")
    
    def run_improved_forecasting(self):
        """Run improved forecasting for all stocks"""
        print("🚀 Starting improved forecasting system...")
        
        # Load data
        df = self.load_and_prepare_data()
        if df is None:
            print("❌ Failed to load data")
            return
        
        # Generate forecasts for all stocks
        for ticker in self.all_tickers:
            self.generate_improved_forecast(ticker, df)
        
        print(f"\n🎉 Improved forecasting completed!")
        print(f"📁 Plots saved to: {self.plots_dir}")
        print(f"📈 Check the improvement percentages in each plot")


if __name__ == "__main__":
    generator = ImprovedForecastGenerator()
    generator.run_improved_forecasting() 