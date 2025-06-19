"""
Prophet model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from prophet import Prophet
from .base_model import BaseForecastModel


class ProphetModel(BaseForecastModel):
    """
    Facebook Prophet model for time series forecasting.
    """
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 **kwargs):
        """
        Initialize Prophet model.
        
        Args:
            yearly_seasonality (bool): Enable yearly seasonality
            weekly_seasonality (bool): Enable weekly seasonality
            daily_seasonality (bool): Enable daily seasonality
            **kwargs: Additional Prophet parameters
        """
        super().__init__("Prophet", 
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        **kwargs)
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        # Initialize Prophet model
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            **kwargs
        )
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for Prophet model.
        
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
        
        # Create Prophet format (ds, y)
        prophet_df = df[[self.date_column, self.target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Remove NaN values
        prophet_df = prophet_df.dropna()
        
        return df, prophet_df
    
    def fit(self, df: pd.DataFrame) -> 'ProphetModel':
        """
        Fit the Prophet model.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            ProphetModel: Self for chaining
        """
        logger.info("Fitting Prophet model")
        
        # Prepare data
        _, prophet_df = self.prepare_data(df)
        
        if len(prophet_df) < 10:
            raise ValueError("Insufficient data for Prophet model")
        
        # Fit Prophet model
        try:
            self.model.fit(prophet_df)
            
            # Store training history
            self.training_history = {
                'n_observations': len(prophet_df),
                'date_range': {
                    'start': prophet_df['ds'].min().strftime('%Y-%m-%d'),
                    'end': prophet_df['ds'].max().strftime('%Y-%m-%d')
                }
            }
            
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
        
        return self
    
    def predict(self, df: pd.DataFrame, periods: int = 30) -> np.ndarray:
        """
        Make predictions with Prophet model.
        
        Args:
            df (pd.DataFrame): Input data
            periods (int): Number of periods to forecast
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Make forecast
        forecast = self.model.predict(future)
        
        # Get predictions for the future periods
        predictions = forecast['yhat'].tail(periods).values
        
        logger.info(f"Prophet forecast generated for {periods} periods")
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
        
        # Prepare data
        _, prophet_df = self.prepare_data(df)
        
        # Make forecast for historical data
        forecast = self.model.predict(prophet_df)
        
        # Get fitted values
        predictions = forecast['yhat'].values
        
        return predictions
    
    def add_regressors(self, df: pd.DataFrame, regressor_columns: list) -> 'ProphetModel':
        """
        Add external regressors to the Prophet model.
        
        Args:
            df (pd.DataFrame): Input data
            regressor_columns (list): List of regressor column names
            
        Returns:
            ProphetModel: Self for chaining
        """
        if self.is_fitted:
            raise ValueError("Cannot add regressors after model is fitted")
        
        # Prepare data
        _, prophet_df = self.prepare_data(df)
        
        # Add regressors
        for col in regressor_columns:
            if col in df.columns:
                prophet_df[col] = df[col].values[:len(prophet_df)]
                self.model.add_regressor(col)
                logger.info(f"Added regressor: {col}")
        
        return self
    
    def plot_components(self, save_path: Optional[str] = None) -> None:
        """
        Plot Prophet model components.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting components")
        
        # Create components plot
        fig = self.model.plot_components(self.model.predict())
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Components plot saved to {save_path}")
        
        return fig
    
    def plot_forecast(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot Prophet forecast.
        
        Args:
            df (pd.DataFrame): Input data
            save_path (Optional[str]): Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting forecast")
        
        # Prepare data
        _, prophet_df = self.prepare_data(df)
        
        # Make forecast
        forecast = self.model.predict(prophet_df)
        
        # Create forecast plot
        fig = self.model.plot(forecast)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        return fig
    
    def cross_validation(self, df: pd.DataFrame, 
                        initial: str = '730 days',
                        period: str = '180 days',
                        horizon: str = '365 days') -> pd.DataFrame:
        """
        Perform cross-validation.
        
        Args:
            df (pd.DataFrame): Input data
            initial (str): Initial training period
            period (str): Period between cutoff dates
            horizon (str): Forecast horizon
            
        Returns:
            pd.DataFrame: Cross-validation results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        # Prepare data
        _, prophet_df = self.prepare_data(df)
        
        # Perform cross-validation
        from prophet.diagnostics import cross_validation, performance_metrics
        
        df_cv = cross_validation(self.model, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        
        logger.info("Cross-validation completed")
        return df_p 