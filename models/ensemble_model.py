"""
Ensemble model that combines multiple forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from .base_model import BaseForecastModel


class EnsembleModel(BaseForecastModel):
    """
    Ensemble model that combines multiple forecasting models.
    """
    
    def __init__(self, 
                 models: List[BaseForecastModel] = None,
                 weights: List[float] = None,
                 ensemble_method: str = 'weighted_average',
                 **kwargs):
        """
        Initialize ensemble model.
        
        Args:
            models (List[BaseForecastModel]): List of models to ensemble
            weights (List[float]): Weights for each model
            ensemble_method (str): Ensemble method ('weighted_average', 'voting', 'stacking')
            **kwargs: Additional parameters
        """
        super().__init__("Ensemble", 
                        ensemble_method=ensemble_method,
                        **kwargs)
        
        self.models = models or []
        self.weights = weights
        self.ensemble_method = ensemble_method
        self.meta_model = None
        
        # Initialize meta-model for stacking
        if ensemble_method == 'stacking':
            self.meta_model = LinearRegression()
    
    def add_model(self, model: BaseForecastModel, weight: float = 1.0) -> 'EnsembleModel':
        """
        Add a model to the ensemble.
        
        Args:
            model (BaseForecastModel): Model to add
            weight (float): Weight for the model
            
        Returns:
            EnsembleModel: Self for chaining
        """
        self.models.append(model)
        
        if self.weights is None:
            self.weights = []
        
        self.weights.append(weight)
        
        logger.info(f"Added {model.name} to ensemble with weight {weight}")
        return self
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for ensemble model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and target
        """
        # For ensemble, we'll use the first model's data preparation
        if not self.models:
            raise ValueError("No models in ensemble")
        
        return self.models[0].prepare_data(df)
    
    def fit(self, df: pd.DataFrame) -> 'EnsembleModel':
        """
        Fit all models in the ensemble.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            EnsembleModel: Self for chaining
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Fitting ensemble with {len(self.models)} models")
        
        # Fit each model
        for model in self.models:
            try:
                model.fit(df)
                logger.info(f"Fitted {model.name}")
            except Exception as e:
                logger.error(f"Error fitting {model.name}: {e}")
                raise
        
        # Fit meta-model if using stacking
        if self.ensemble_method == 'stacking':
            self._fit_meta_model(df)
        
        self.is_fitted = True
        logger.info("Ensemble model fitted successfully")
        
        return self
    
    def _fit_meta_model(self, df: pd.DataFrame) -> None:
        """
        Fit meta-model for stacking ensemble.
        
        Args:
            df (pd.DataFrame): Training data
        """
        # Get predictions from all models
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict_in_sample(df)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Could not get predictions from {model.name}: {e}")
                continue
        
        if len(predictions) < 2:
            raise ValueError("Need at least 2 models for stacking")
        
        # Prepare meta-features
        meta_features = np.column_stack(predictions)
        
        # Get true values
        _, target_series = self.prepare_data(df)
        y_true = target_series.values
        
        # Remove NaN values
        mask = ~np.isnan(y_true)
        meta_features_clean = meta_features[mask]
        y_true_clean = y_true[mask]
        
        # Fit meta-model
        self.meta_model.fit(meta_features_clean, y_true_clean)
        
        logger.info("Meta-model fitted successfully")
    
    def predict(self, df: pd.DataFrame, steps: int = 30) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            df (pd.DataFrame): Input data
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        predictions = []
        valid_models = []
        
        for model in self.models:
            try:
                pred = model.predict(df, steps)
                predictions.append(pred)
                valid_models.append(model)
                logger.info(f"Got predictions from {model.name}")
            except Exception as e:
                logger.warning(f"Could not get predictions from {model.name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average(predictions, valid_models)
        elif self.ensemble_method == 'voting':
            return self._voting(predictions)
        elif self.ensemble_method == 'stacking':
            return self._stacking(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_average(self, predictions: List[np.ndarray], 
                         valid_models: List[BaseForecastModel]) -> np.ndarray:
        """
        Combine predictions using weighted average.
        
        Args:
            predictions (List[np.ndarray]): List of predictions
            valid_models (List[BaseForecastModel]): List of valid models
            
        Returns:
            np.ndarray: Weighted average predictions
        """
        # Get weights for valid models
        valid_weights = []
        for model in valid_models:
            model_idx = self.models.index(model)
            valid_weights.append(self.weights[model_idx])
        
        # Normalize weights
        total_weight = sum(valid_weights)
        normalized_weights = [w / total_weight for w in valid_weights]
        
        # Calculate weighted average
        weighted_pred = np.zeros_like(predictions[0])
        
        for pred, weight in zip(predictions, normalized_weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def _voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions using voting (median).
        
        Args:
            predictions (List[np.ndarray]): List of predictions
            
        Returns:
            np.ndarray: Median predictions
        """
        return np.median(predictions, axis=0)
    
    def _stacking(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions using stacking.
        
        Args:
            predictions (List[np.ndarray]): List of predictions
            
        Returns:
            np.ndarray: Stacked predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not fitted")
        
        # Stack predictions
        meta_features = np.column_stack(predictions)
        
        # Make meta-predictions
        stacked_pred = self.meta_model.predict(meta_features)
        
        return stacked_pred
    
    def predict_in_sample(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make in-sample ensemble predictions.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: In-sample ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get in-sample predictions from all models
        predictions = []
        valid_models = []
        
        for model in self.models:
            try:
                pred = model.predict_in_sample(df)
                predictions.append(pred)
                valid_models.append(model)
            except Exception as e:
                logger.warning(f"Could not get in-sample predictions from {model.name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid in-sample predictions from any model")
        
        # Combine predictions
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average(predictions, valid_models)
        elif self.ensemble_method == 'voting':
            return self._voting(predictions)
        elif self.ensemble_method == 'stacking':
            return self._stacking(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current model weights.
        
        Returns:
            Dict[str, float]: Model weights
        """
        if not self.weights:
            return {}
        
        return dict(zip([model.name for model in self.models], self.weights))
    
    def set_model_weights(self, weights: Dict[str, float]) -> 'EnsembleModel':
        """
        Set model weights.
        
        Args:
            weights (Dict[str, float]): Model weights
            
        Returns:
            EnsembleModel: Self for chaining
        """
        self.weights = []
        
        for model in self.models:
            weight = weights.get(model.name, 1.0)
            self.weights.append(weight)
        
        logger.info(f"Updated model weights: {weights}")
        return self
    
    def evaluate_individual_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each model individually.
        
        Args:
            df (pd.DataFrame): Test data
            
        Returns:
            Dict[str, Dict[str, float]]: Individual model metrics
        """
        results = {}
        
        for model in self.models:
            try:
                metrics = model.evaluate(df)
                results[model.name] = metrics
                logger.info(f"{model.name} metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error evaluating {model.name}: {e}")
                results[model.name] = {}
        
        return results 