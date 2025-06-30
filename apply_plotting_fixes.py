"""
Apply Plotting Fixes to Existing Forecasting System
This script demonstrates how the plotting and data validation fixes
can resolve the shape mismatch and visualization issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our fixes
from utils.plotting_fixes import ForecastDataValidator, RobustForecasterPlotter, quick_fix_forecast_data

# Import existing models
from models.lstm_model import LSTMModel
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel

# Import configuration
from config import PROCESSED_DATA_DIR, MODELS_DIR


class FixedForecastEvaluator:
    """Enhanced forecast evaluator that uses robust data validation and plotting"""
    
    def __init__(self):
        self.data_validator = ForecastDataValidator()
        self.plotter = RobustForecasterPlotter()
        self.results = {}
        
    def load_test_data(self, ticker: str = 'AAPL') -> pd.DataFrame:
        """Load test data for demonstration"""
        # Try to load actual data
        try:
            data_file = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter for specific ticker if available
                if 'ticker' in df.columns:
                    ticker_data = df[df['ticker'] == ticker]
                    if len(ticker_data) > 100:
                        return ticker_data.sort_values('date').reset_index(drop=True)
                
                # Use first ticker if specified ticker not found
                if 'ticker' in df.columns:
                    first_ticker = df['ticker'].iloc[0]
                    ticker_data = df[df['ticker'] == first_ticker]
                    logger.info(f"Using {first_ticker} data instead of {ticker}")
                    return ticker_data.sort_values('date').reset_index(drop=True)
                
                # Use all data if no ticker column
                return df.sort_values('date').reset_index(drop=True)
                
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
        
        # Generate synthetic data for demonstration
        logger.info("Generating synthetic test data")
        dates = pd.date_range('2020-01-01', '2024-06-01', freq='D')
        
        # Generate realistic stock price data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Add some trend and seasonality
        trend = np.linspace(0, 0.5, len(dates))
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Annual seasonality
        prices = prices * (1 + trend + seasonal)
        
        df = pd.DataFrame({
            'date': dates,
            'close_price': prices,
            'ticker': ticker
        })
        
        return df
    
    def create_problematic_forecasts(self, actual_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create forecasts with typical shape/data issues to demonstrate fixes"""
        
        test_period = actual_data['date'] >= '2024-01-01'
        test_data = actual_data[test_period]
        actual_prices = test_data['close_price'].values
        
        # Create various problematic forecast formats
        forecasts = {}
        
        # 1. Correct 1D forecast
        forecasts['Good_1D'] = actual_prices + np.random.normal(0, 5, len(actual_prices))
        
        # 2. 2D forecast (n, 1) shape - common LSTM output issue
        forecast_2d = (actual_prices + np.random.normal(0, 8, len(actual_prices))).reshape(-1, 1)
        forecasts['Bad_2D_Shape'] = forecast_2d
        
        # 3. Forecast with NaN values - common preprocessing issue  
        forecast_with_nans = actual_prices + np.random.normal(0, 6, len(actual_prices))
        forecast_with_nans[10:15] = np.nan
        forecast_with_nans[50:52] = np.inf
        forecasts['With_NaN_Inf'] = forecast_with_nans
        
        # 4. Wrong length forecast - common array mismatch
        wrong_length = actual_prices[:-5] + np.random.normal(0, 7, len(actual_prices) - 5)
        forecasts['Wrong_Length'] = wrong_length
        
        # 5. Multiple columns (ensemble output) - transpose issue
        multi_col = np.column_stack([
            actual_prices + np.random.normal(0, 4, len(actual_prices)),
            actual_prices + np.random.normal(0, 6, len(actual_prices)),
            actual_prices + np.random.normal(0, 5, len(actual_prices))
        ])
        forecasts['Multi_Column'] = multi_col
        
        # 6. Extreme outliers - data quality issue
        with_outliers = actual_prices + np.random.normal(0, 5, len(actual_prices))
        with_outliers[25] = actual_prices[25] * 10  # Extreme outlier
        with_outliers[75] = -1000  # Another extreme outlier
        forecasts['With_Outliers'] = with_outliers
        
        return forecasts, test_data
    
    def demonstrate_issues_and_fixes(self, ticker: str = 'AAPL'):
        """Demonstrate the plotting issues and how fixes resolve them"""
        
        logger.info(f"=== DEMONSTRATING PLOTTING FIXES FOR {ticker} ===")
        
        # Load test data
        df = self.load_test_data(ticker)
        
        # Create problematic forecasts
        forecasts, test_data = self.create_problematic_forecasts(df)
        
        dates = test_data['date'].values
        actual_values = test_data['close_price'].values
        
        logger.info(f"Test data: {len(dates)} dates, {len(actual_values)} actual values")
        
        # === BEFORE FIXES: Show the problems ===
        logger.info("\n=== BEFORE FIXES: Attempting to plot with raw data ===")
        
        for name, forecast in forecasts.items():
            try:
                logger.info(f"\nTrying to plot {name}:")
                logger.info(f"  Forecast shape: {np.array(forecast).shape}")
                logger.info(f"  Dates shape: {np.array(dates).shape}")
                logger.info(f"  Contains NaN: {np.any(np.isnan(forecast)) if np.array(forecast).dtype.kind in 'fc' else 'No'}")
                logger.info(f"  Contains Inf: {np.any(np.isinf(forecast)) if np.array(forecast).dtype.kind in 'fc' else 'No'}")
                
                # Try basic plotting (this would fail with shape mismatches)
                if len(dates) != len(np.array(forecast).flatten()):
                    logger.error(f"  ❌ LENGTH MISMATCH: dates={len(dates)}, forecast={len(np.array(forecast).flatten())}")
                else:
                    logger.info(f"  ✅ Lengths match")
                    
            except Exception as e:
                logger.error(f"  ❌ Would fail with error: {e}")
        
        # === AFTER FIXES: Show the solutions ===
        logger.info(f"\n{'='*60}")
        logger.info("=== AFTER FIXES: Using robust data validation ===")
        logger.info(f"{'='*60}")
        
        fixed_forecasts = {}
        metrics_summary = {}
        
        for name, forecast in forecasts.items():
            try:
                logger.info(f"\nFixing {name}:")
                
                # Apply data validation and fixes
                fixed_dates, fixed_forecast, fixed_actual = self.data_validator.validate_and_fix_forecast_data(
                    dates=dates,
                    forecasts=forecast,
                    actual_values=actual_values,
                    forecast_name=name
                )
                
                fixed_forecasts[name] = fixed_forecast
                
                # Calculate robust metrics
                metrics = self.plotter.calculate_robust_metrics(
                    actual=fixed_actual, 
                    predicted=fixed_forecast,
                    model_name=name
                )
                metrics_summary[name] = metrics
                
                logger.info(f"  ✅ Successfully fixed and validated {name}")
                logger.info(f"  Final shapes: dates={len(fixed_dates)}, forecast={len(fixed_forecast)}")
                
            except Exception as e:
                logger.error(f"  ❌ Could not fix {name}: {e}")
        
        # === CREATE COMPARISON PLOTS ===
        logger.info(f"\n{'='*60}")
        logger.info("=== CREATING ROBUST COMPARISON PLOTS ===")
        logger.info(f"{'='*60}")
        
        if fixed_forecasts:
            # Create comprehensive comparison plot
            fig = self.plotter.plot_forecast_comparison(
                dates=dates,
                actual_values=actual_values,
                forecasts=fixed_forecasts,
                title=f"Robust Forecast Comparison: {ticker} (Data Validation Fixes Applied)",
                save_path=f"robust_forecast_comparison_{ticker}.png"
            )
            
            logger.info(f"✅ Robust comparison plot saved successfully")
            
            # Show metrics summary
            logger.info(f"\n{'='*40}")
            logger.info("METRICS SUMMARY (After Fixes)")
            logger.info(f"{'='*40}")
            
            for model_name, metrics in metrics_summary.items():
                if metrics:
                    logger.info(f"\n{model_name}:")
                    for metric, value in metrics.items():
                        logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return fixed_forecasts, metrics_summary
    
    def test_with_real_models(self, ticker: str = 'AAPL'):
        """Test the fixes with actual model predictions"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"=== TESTING WITH REAL MODELS: {ticker} ===")
        logger.info(f"{'='*60}")
        
        df = self.load_test_data(ticker)
        
        # Split data
        train_end = '2023-12-31'
        train_data = df[df['date'] <= train_end].copy()
        test_data = df[df['date'] > train_end].copy()
        
        if len(test_data) < 10:
            logger.warning("Not enough test data for real model testing")
            return
        
        logger.info(f"Train data: {len(train_data)} samples")
        logger.info(f"Test data: {len(test_data)} samples")
        
        # Initialize models
        models = {}
        
        try:
            # Prophet model
            prophet_model = ProphetModel()
            prophet_model.date_column = 'date'
            prophet_model.target_column = 'close_price'
            
            # Prepare Prophet data
            prophet_train = train_data[['date', 'close_price']].rename(
                columns={'date': 'ds', 'close_price': 'y'}
            )
            prophet_model.date_column = 'ds'
            prophet_model.target_column = 'y'
            
            prophet_model.fit(prophet_train)
            prophet_forecast = prophet_model.predict(prophet_train, periods=len(test_data))
            
            models['Prophet'] = prophet_forecast
            logger.info("✅ Prophet model trained and predicted")
            
        except Exception as e:
            logger.error(f"❌ Prophet model failed: {e}")
        
        try:
            # ARIMA model
            arima_model = ARIMAModel()
            arima_model.date_column = 'date'
            arima_model.target_column = 'close_price'
            
            arima_model.fit(train_data)
            arima_forecast = arima_model.predict(train_data, steps=len(test_data))
            
            models['ARIMA'] = arima_forecast
            logger.info("✅ ARIMA model trained and predicted")
            
        except Exception as e:
            logger.error(f"❌ ARIMA model failed: {e}")
        
        # Test plotting with real model outputs
        if models:
            actual_values = test_data['close_price'].values
            dates = test_data['date'].values
            
            logger.info("\n=== Testing real model predictions with robust plotting ===")
            
            fig = self.plotter.plot_forecast_comparison(
                dates=dates,
                actual_values=actual_values,
                forecasts=models,
                title=f"Real Model Forecasts with Robust Plotting: {ticker}",
                save_path=f"real_model_robust_comparison_{ticker}.png"
            )
            
            # Calculate metrics for real models
            for model_name, forecast in models.items():
                metrics = self.plotter.calculate_robust_metrics(
                    actual=actual_values,
                    predicted=forecast,
                    model_name=model_name
                )
                logger.info(f"\n{model_name} Metrics:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return models


def main():
    """Main function to demonstrate plotting fixes"""
    
    # Setup logging
    logger.add("logs/plotting_fixes_demo.log", rotation="10 MB")
    
    logger.info("="*80)
    logger.info("FORECAST PLOTTING FIXES DEMONSTRATION")
    logger.info("="*80)
    
    evaluator = FixedForecastEvaluator()
    
    # Test with problematic synthetic data
    logger.info("Phase 1: Testing with problematic synthetic forecasts...")
    fixed_forecasts, metrics = evaluator.demonstrate_issues_and_fixes('TEST_STOCK')
    
    # Test with real models if possible
    logger.info("\nPhase 2: Testing with real model predictions...")
    real_models = evaluator.test_with_real_models('TEST_STOCK')
    
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nKey Fixes Applied:")
    logger.info("✅ Shape mismatch resolution (2D -> 1D)")
    logger.info("✅ Length alignment (truncate to minimum)")
    logger.info("✅ NaN/Inf value cleaning")
    logger.info("✅ Extreme outlier detection and removal")
    logger.info("✅ Robust metric calculation")
    logger.info("✅ Consistent plot formatting")
    
    logger.info(f"\nCheck the generated plots:")
    logger.info("- robust_forecast_comparison_TEST_STOCK.png")
    logger.info("- real_model_robust_comparison_TEST_STOCK.png")


if __name__ == "__main__":
    main() 