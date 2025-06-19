"""
Base model class for all forecasting models.
Provides common interface and functionality.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import joblib
from pathlib import Path
import json


class BaseForecastModel(ABC):
    """
    Abstract base class for all forecasting models.
    Provides common interface and functionality.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            name (str): Model name
            **kwargs: Additional model parameters
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        self.target_column = 'close_price'
        self.date_column = 'date'
        self.model_params = kwargs
        self.training_history = {}
        
        logger.info(f"Initialized {self.name} model with parameters: {kwargs}")
    
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training/prediction.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target
        """
        pass
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BaseForecastModel':
        """
        Fit the model to the data.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            BaseForecastModel: Self for chaining
        """
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            df (pd.DataFrame): Test data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Prepare data
        X, y_true = self.prepare_data(df)
        
        # Make predictions
        y_pred = self.predict(df)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        
        logger.info(f"{self.name} evaluation metrics: {metrics}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'mape': np.nan,
                'r2': np.nan
            }
        
        # Calculate metrics
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        mse = np.mean((y_true_clean - y_pred_clean) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        
        # R-squared
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"{filepath}.joblib")
        
        # Save metadata
        metadata = {
            'name': self.name,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'date_column': self.date_column,
            'model_params': self.model_params,
            'training_history': self.training_history
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'BaseForecastModel':
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            BaseForecastModel: Self for chaining
        """
        # Load model
        self.model = joblib.load(f"{filepath}.joblib")
        
        # Load metadata
        with open(f"{filepath}.json", 'r') as f:
            metadata = json.load(f)
        
        self.name = metadata['name']
        self.feature_columns = metadata['feature_columns']
        self.target_column = metadata['target_column']
        self.date_column = metadata['date_column']
        self.model_params = metadata['model_params']
        self.training_history = metadata.get('training_history', {})
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Optional[Dict[str, float]]: Feature importance scores
        """
        if not self.is_fitted or not hasattr(self.model, 'feature_importances_'):
            return None
        
        return dict(zip(self.feature_columns, self.model.feature_importances_))
    
    def __str__(self) -> str:
        return f"{self.name} (fitted: {self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__() 