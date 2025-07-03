"""
ARIMA model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from .base_model import BaseForecastModel
import pmdarima as pm


class ARIMAModel(BaseForecastModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), **kwargs):
        """
        Initialize ARIMA model.
        
        Args:
            order (Tuple[int, int, int]): ARIMA order (p, d, q)
            **kwargs: Additional parameters
        """
        super().__init__("ARIMA", order=order, **kwargs)
        self.order = order
        self.fitted_model = None
        self.is_stationary = False
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for ARIMA model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target
        """
        # Ensure date column is datetime
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        df = df.sort_values(self.date_column)
        
        # Set date as index
        df = df.set_index(self.date_column)
        
        # Resample to business day frequency to ensure regularity
        df = df.asfreq('B').ffill()

        # Get target series
        target_series = df[self.target_column].dropna()
        
        # Check for stationarity
        self._check_stationarity(target_series)
        
        return df, target_series
    
    def _check_stationarity(self, series: pd.Series) -> bool:
        """
        Check if the time series is stationary.
        
        Args:
            series (pd.Series): Time series to check
            
        Returns:
            bool: True if stationary
        """
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series.dropna())
        
        # Extract p-value
        p_value = result[1]
        
        self.is_stationary = p_value < 0.05
        
        logger.info(f"Stationarity test p-value: {p_value:.4f}")
        logger.info(f"Series is {'stationary' if self.is_stationary else 'non-stationary'}")
        
        return self.is_stationary
    
    def _auto_determine_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Automatically determine ARIMA order using AIC.
        
        Args:
            series (pd.Series): Time series
            
        Returns:
            Tuple[int, int, int]: Optimal ARIMA order
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Grid search for optimal order
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        logger.info(f"Auto-determined ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit(self, df: pd.DataFrame) -> 'ARIMAModel':
        """
        Fit the ARIMA model.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            ARIMAModel: Self for chaining
        """
        logger.info(f"Fitting ARIMA model with order {self.order}")
        
        # Prepare data
        _, target_series = self.prepare_data(df)
        
        if len(target_series) < 30: # Increased required data length
            logger.warning("Insufficient data for ARIMA model (< 30 records)")
            return self

        try:
            # Use auto_arima to find the best model
            automodel = pm.auto_arima(
                target_series,
                start_p=1, start_q=1,
                test='adf',
                max_p=5, max_q=5,
                m=1, # Non-seasonal
                d=None, # Let the model determine 'd'
                seasonal=False,
                start_P=0,
                D=0,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            self.fitted_model = automodel
            self.order = automodel.order
            
            self.training_history = {
                'aic': self.fitted_model.aic(),
                'bic': self.fitted_model.bic(),
                'order': self.order,
                'n_observations': len(target_series)
            }
            
            self.is_fitted = True
            logger.info(f"ARIMA model fitted successfully with order {self.order}. AIC: {self.fitted_model.aic():.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting auto_arima model: {str(e)}")
            # Fallback to a simple ARIMA if auto_arima fails
            try:
                logger.warning("auto_arima failed. Falling back to default order (1,1,1).")
                self.order = (1, 1, 1)
                model = ARIMA(target_series, order=self.order)
                self.fitted_model = model.fit()
                self.is_fitted = True
                logger.info("Fallback ARIMA model fitted successfully.")
            except Exception as fallback_e:
                logger.error(f"Fallback ARIMA also failed: {str(fallback_e)}")
                raise fallback_e
        
        return self
    
    def predict(self, df: pd.DataFrame, steps: int = 30) -> np.ndarray:
        """
        Make predictions with ARIMA model.
        
        Args:
            df (pd.DataFrame): Input data
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare data
        _, target_series = self.prepare_data(df)
        
        # Make forecast
        forecast = self.fitted_model.predict(n_periods=steps)
        
        # Convert to numpy array
        predictions = np.array(forecast)
        
        logger.info(f"ARIMA forecast generated for {steps} steps")
        return predictions
    
    def predict_in_sample(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make in-sample predictions.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: In-sample predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = self.fitted_model.predict_in_sample()
        
        return predictions
    
    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """
        Plot ARIMA model diagnostics.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        self.fitted_model.plot_diagnostics(figsize=(12, 8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diagnostics plot saved to {save_path}")
        
        plt.show()
    
    def get_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            str: Model summary
        """
        if not self.is_fitted:
            return "Model not fitted"
        
        return self.fitted_model.summary().as_text()
    
    def save_model(self, file_path: str):
        # Implementation of save_model method
        pass 