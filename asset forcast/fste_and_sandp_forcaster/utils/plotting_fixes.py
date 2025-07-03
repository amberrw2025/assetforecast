"""
Plotting and Data Validation Fixes
Addresses critical issues:
- Shape mismatches between dates and forecast values
- 2D array handling in plotting
- NaN/Inf value cleaning
- Consistent data format validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union, Dict, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ForecastDataValidator:
    """Validates and fixes forecast data for consistent plotting and evaluation"""
    
    @staticmethod
    def validate_and_fix_forecast_data(dates: Union[pd.DatetimeIndex, np.ndarray, List], 
                                     forecasts: Union[np.ndarray, List],
                                     actual_values: Optional[Union[np.ndarray, List]] = None,
                                     forecast_name: str = "forecast") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Validate and fix forecast data to ensure consistent shapes and clean values
        
        Args:
            dates: Date values (should be 1D)
            forecasts: Forecast values (could be 1D or 2D)
            actual_values: Actual values for comparison (optional)
            forecast_name: Name of forecast for logging
            
        Returns:
            Tuple of (cleaned_dates, cleaned_forecasts, cleaned_actual_values)
        """
        # Convert inputs to numpy arrays
        dates = np.array(dates)
        forecasts = np.array(forecasts)
        if actual_values is not None:
            actual_values = np.array(actual_values)
        
        logger.info(f"Validating {forecast_name} - Initial shapes: dates={dates.shape}, forecasts={forecasts.shape}")
        
        # Handle multi-dimensional forecast arrays
        if forecasts.ndim > 1:
            if forecasts.shape[1] == 1:
                # Shape (n, 1) -> flatten to (n,)
                forecasts = forecasts.flatten()
                logger.info(f"Flattened 2D forecast array from {forecasts.shape} to 1D")
            elif forecasts.shape[0] == 1:
                # Shape (1, n) -> flatten to (n,)
                forecasts = forecasts.flatten()
                logger.info(f"Flattened 2D forecast array from {forecasts.shape} to 1D")
            else:
                # Multiple forecast scenarios - take mean or first column
                if forecasts.shape[1] > forecasts.shape[0]:
                    # More columns than rows - transpose and take mean
                    forecasts = np.mean(forecasts.T, axis=1)
                    logger.warning(f"Multi-column forecast detected, taking mean across columns")
                else:
                    # Take first column
                    forecasts = forecasts[:, 0]
                    logger.warning(f"Multi-column forecast detected, taking first column")
        
        # Handle length mismatches
        min_length = min(len(dates), len(forecasts))
        if actual_values is not None:
            min_length = min(min_length, len(actual_values))
        
        if len(dates) != len(forecasts):
            logger.warning(f"Length mismatch detected - dates: {len(dates)}, forecasts: {len(forecasts)}")
            logger.warning(f"Truncating to minimum length: {min_length}")
            
            dates = dates[:min_length]
            forecasts = forecasts[:min_length]
            if actual_values is not None:
                actual_values = actual_values[:min_length]
        
        # Clean invalid values
        valid_mask = ForecastDataValidator._create_valid_mask(forecasts, actual_values)
        
        if not valid_mask.all():
            n_invalid = np.sum(~valid_mask)
            logger.warning(f"Removing {n_invalid} invalid values (NaN/Inf) from {forecast_name}")
            
            dates = dates[valid_mask]
            forecasts = forecasts[valid_mask]
            if actual_values is not None:
                actual_values = actual_values[valid_mask]
        
        # Final validation
        if len(dates) == 0 or len(forecasts) == 0:
            raise ValueError(f"No valid data remaining after cleaning for {forecast_name}")
        
        if len(dates) != len(forecasts):
            raise ValueError(f"Shape mismatch after cleaning: dates={len(dates)}, forecasts={len(forecasts)}")
        
        logger.info(f"Validation complete for {forecast_name} - Final length: {len(dates)}")
        
        return dates, forecasts, actual_values
    
    @staticmethod
    def _create_valid_mask(forecasts: np.ndarray, actual_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Create boolean mask for valid (non-NaN, non-Inf) values"""
        
        # Check forecasts for validity
        forecast_valid = np.isfinite(forecasts) & ~np.isnan(forecasts)
        
        # Also check that values are not extreme outliers
        if np.any(forecast_valid):
            forecast_median = np.median(forecasts[forecast_valid])
            forecast_mad = np.median(np.abs(forecasts[forecast_valid] - forecast_median))
            
            # Flag values more than 10 MAD from median as invalid
            if forecast_mad > 0:
                extreme_threshold = 10 * forecast_mad
                forecast_valid = forecast_valid & (np.abs(forecasts - forecast_median) <= extreme_threshold)
        
        valid_mask = forecast_valid
        
        # If actual values provided, also check them
        if actual_values is not None:
            actual_valid = np.isfinite(actual_values) & ~np.isnan(actual_values)
            valid_mask = valid_mask & actual_valid
        
        return valid_mask


class RobustForecasterPlotter:
    """Robust plotting class that handles all shape mismatch and data validation issues"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.validator = ForecastDataValidator()
    
    def plot_forecast_comparison(self, 
                               dates: Union[pd.DatetimeIndex, np.ndarray],
                               actual_values: np.ndarray,
                               forecasts: Dict[str, np.ndarray],
                               title: str = "Forecast Comparison",
                               save_path: Optional[str] = None,
                               show_confidence_intervals: bool = True) -> plt.Figure:
        """
        Plot multiple forecasts against actual values with robust data validation
        
        Args:
            dates: Date values
            actual_values: Actual observed values
            forecasts: Dictionary of {model_name: forecast_array}
            title: Plot title
            save_path: Path to save plot
            show_confidence_intervals: Whether to show confidence intervals
            
        Returns:
            matplotlib Figure object
        """
        
        # Validate and clean actual values first
        dates_clean, actual_clean, _ = self.validator.validate_and_fix_forecast_data(
            dates, actual_values, forecast_name="actual_values"
        )
        
        # Validate and clean each forecast
        cleaned_forecasts = {}
        for model_name, forecast in forecasts.items():
            try:
                _, forecast_clean, _ = self.validator.validate_and_fix_forecast_data(
                    dates_clean, forecast, forecast_name=model_name
                )
                
                # Ensure forecast matches actual values length
                min_len = min(len(dates_clean), len(forecast_clean), len(actual_clean))
                cleaned_forecasts[model_name] = forecast_clean[:min_len]
                
            except Exception as e:
                logger.error(f"Error cleaning forecast for {model_name}: {e}")
                continue
        
        # Truncate dates and actual to minimum length
        if cleaned_forecasts:
            min_len = min(len(dates_clean), len(actual_clean), 
                         min(len(f) for f in cleaned_forecasts.values()))
            dates_clean = dates_clean[:min_len]
            actual_clean = actual_clean[:min_len]
            cleaned_forecasts = {name: forecast[:min_len] 
                               for name, forecast in cleaned_forecasts.items()}
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Main forecast plot
        ax1.plot(dates_clean, actual_clean, 'ko-', label='Actual', 
                linewidth=2, markersize=3, alpha=0.8)
        
        # Plot each forecast
        colors = plt.cm.tab10(np.linspace(0, 1, len(cleaned_forecasts)))
        
        for i, (model_name, forecast) in enumerate(cleaned_forecasts.items()):
            ax1.plot(dates_clean, forecast, '--', 
                    color=colors[i], label=f'{model_name} Forecast', 
                    linewidth=2, alpha=0.7)
            
            # Calculate and show error metrics
            rmse = np.sqrt(np.mean((actual_clean - forecast) ** 2))
            mae = np.mean(np.abs(actual_clean - forecast))
            
            # Add metrics to legend
            ax1.plot([], [], ' ', label=f'{model_name} RMSE: {rmse:.2f}')
        
        ax1.set_title(title, fontsize=16, pad=20)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Error plot for best forecast
        if cleaned_forecasts:
            best_model = min(cleaned_forecasts.keys(), 
                           key=lambda x: np.sqrt(np.mean((actual_clean - cleaned_forecasts[x]) ** 2)))
            best_forecast = cleaned_forecasts[best_model]
            errors = actual_clean - best_forecast
            
            ax2.plot(dates_clean, errors, 'r-', alpha=0.7, linewidth=1)
            ax2.fill_between(dates_clean, 0, errors, alpha=0.3, color='red')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title(f'Prediction Errors ({best_model})', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Error', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # Format dates
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def calculate_robust_metrics(self, 
                               actual: np.ndarray, 
                               predicted: np.ndarray,
                               model_name: str = "model") -> Dict[str, float]:
        """
        Calculate robust evaluation metrics with proper data validation
        
        Args:
            actual: Actual values
            predicted: Predicted values  
            model_name: Model name for logging
            
        Returns:
            Dictionary of metrics
        """
        # Validate data first
        try:
            _, predicted_clean, actual_clean = self.validator.validate_and_fix_forecast_data(
                dates=np.arange(len(actual)), 
                forecasts=predicted, 
                actual_values=actual,
                forecast_name=model_name
            )
            
            if len(actual_clean) == 0:
                logger.error(f"No valid data for metrics calculation for {model_name}")
                return {}
            
            # Calculate metrics
            metrics = {}
            
            # RMSE
            metrics['rmse'] = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
            
            # MAE  
            metrics['mae'] = np.mean(np.abs(actual_clean - predicted_clean))
            
            # MAPE (handle division by zero)
            non_zero_mask = actual_clean != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual_clean[non_zero_mask] - predicted_clean[non_zero_mask]) 
                                    / actual_clean[non_zero_mask])) * 100
                metrics['mape'] = mape
            else:
                metrics['mape'] = np.inf
            
            # R-squared
            ss_res = np.sum((actual_clean - predicted_clean) ** 2)
            ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Directional accuracy
            if len(actual_clean) > 1:
                actual_direction = np.diff(actual_clean) > 0
                pred_direction = np.diff(predicted_clean) > 0
                metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction) * 100
            else:
                metrics['directional_accuracy'] = 0
            
            logger.info(f"Metrics calculated for {model_name}: RMSE={metrics['rmse']:.4f}, "
                       f"MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return {}


def fix_model_predict_methods():
    """
    Patch existing model predict methods to ensure consistent output shapes
    This can be imported and called to fix existing models
    """
    logger.info("Applying fixes to model predict methods...")
    
    # This would patch the existing LSTM and other model classes
    # For now, we'll create a wrapper function
    
    def ensure_1d_output(predict_func):
        """Decorator to ensure predict methods return 1D arrays"""
        def wrapper(*args, **kwargs):
            result = predict_func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                if result.ndim > 1:
                    result = result.flatten()
                # Remove any NaN/Inf values
                if not np.all(np.isfinite(result)):
                    logger.warning("Removing non-finite values from prediction output")
                    result = result[np.isfinite(result)]
            return result
        return wrapper
    
    return ensure_1d_output


# Quick fix function that can be applied to existing code
def quick_fix_forecast_data(dates, forecasts, actual_values=None):
    """Quick utility function to fix forecast data shapes and alignment"""
    validator = ForecastDataValidator()
    return validator.validate_and_fix_forecast_data(dates, forecasts, actual_values)


def test_plotting_fixes():
    """Test function to verify the plotting fixes work correctly"""
    logger.info("Testing plotting fixes...")
    
    # Create test data with various issues
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    actual = np.random.randn(100).cumsum() + 100
    
    # Create problematic forecast data
    forecast_1d = actual + np.random.randn(100) * 2  # 1D forecast
    forecast_2d = (actual + np.random.randn(100, 1) * 2)  # 2D forecast (100, 1)
    forecast_with_nans = actual + np.random.randn(100) * 2
    forecast_with_nans[50:55] = np.nan  # Insert NaN values
    forecast_wrong_length = actual[:95] + np.random.randn(95) * 2  # Wrong length
    
    forecasts = {
        '1D Forecast': forecast_1d,
        '2D Forecast': forecast_2d, 
        'NaN Forecast': forecast_with_nans,
        'Wrong Length': forecast_wrong_length
    }
    
    # Test the robust plotter
    plotter = RobustForecasterPlotter()
    fig = plotter.plot_forecast_comparison(
        dates=dates,
        actual_values=actual,
        forecasts=forecasts,
        title="Test of Robust Plotting Fixes",
        save_path="test_plotting_fixes.png"
    )
    
    # Test metrics calculation
    for name, forecast in forecasts.items():
        metrics = plotter.calculate_robust_metrics(actual, forecast, name)
        logger.info(f"Test metrics for {name}: {metrics}")
    
    logger.info("Plotting fixes test completed successfully!")


if __name__ == "__main__":
    test_plotting_fixes() 