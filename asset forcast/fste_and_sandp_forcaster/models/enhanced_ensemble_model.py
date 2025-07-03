"""
Enhanced Ensemble Model with Historical Accuracy-Based Weighting and Confidence Intervals
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from .base_model import BaseForecastModel


class EnhancedEnsembleModel(BaseForecastModel):
    """
    Enhanced ensemble model with dynamic weighting based on historical accuracy
    and confidence interval estimation.
    """
    
    def __init__(self, 
                 models: List[BaseForecastModel] = None,
                 initial_weights: List[float] = None,
                 weighting_method: str = 'performance_based',
                 performance_window: int = 30,
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 1000,
                 **kwargs):
        """
        Initialize enhanced ensemble model.
        
        Args:
            models: List of models to ensemble
            initial_weights: Initial weights for models
            weighting_method: 'equal', 'performance_based', 'adaptive'
            performance_window: Number of recent predictions for weight calculation
            confidence_level: Confidence level for prediction intervals (0.8, 0.9, 0.95, 0.99)
            bootstrap_samples: Number of bootstrap samples for confidence intervals
        """
        super().__init__("Enhanced Ensemble", **kwargs)
        
        self.models = models or []
        self.initial_weights = initial_weights
        self.weighting_method = weighting_method
        self.performance_window = performance_window
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        
        # Performance tracking
        self.model_performance_history = {model.name: [] for model in self.models}
        self.prediction_history = []
        self.actual_history = []
        self.current_weights = {}
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        if not self.models:
            return
        
        if self.initial_weights:
            total_weight = sum(self.initial_weights)
            self.current_weights = {
                model.name: weight / total_weight 
                for model, weight in zip(self.models, self.initial_weights)
            }
        else:
            # Equal weights initially
            equal_weight = 1.0 / len(self.models)
            self.current_weights = {model.name: equal_weight for model in self.models}
    
    def add_model(self, model: BaseForecastModel, weight: float = None) -> 'EnhancedEnsembleModel':
        """Add a model to the ensemble."""
        self.models.append(model)
        self.model_performance_history[model.name] = []
        
        if weight is None:
            weight = 1.0 / len(self.models)
        
        # Rebalance weights
        self._initialize_weights()
        
        logger.info(f"Added {model.name} to enhanced ensemble")
        return self
    
    def calculate_performance_metrics(self, predictions: np.ndarray, 
                                    actuals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_clean = predictions[mask]
        actual_clean = actuals[mask]
        
        if len(pred_clean) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        rmse = np.sqrt(mean_squared_error(actual_clean, pred_clean))
        mae = mean_absolute_error(actual_clean, pred_clean)
        r2 = r2_score(actual_clean, pred_clean)
        
        # Directional accuracy
        actual_direction = np.diff(actual_clean) > 0
        pred_direction = np.diff(pred_clean) > 0
        
        if len(actual_direction) > 0:
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def update_model_weights(self, model_predictions: Dict[str, np.ndarray], 
                           actuals: np.ndarray):
        """Update model weights based on recent performance."""
        
        if self.weighting_method == 'equal':
            return  # No weight updates for equal weighting
        
        # Calculate performance for each model
        model_scores = {}
        
        for model_name, predictions in model_predictions.items():
            # Align actuals with predictions. Assumes predictions align with the end of the actuals array.
            if len(predictions) != len(actuals):
                logger.debug(f"Aligning actuals (len {len(actuals)}) with predictions for {model_name} (len {len(predictions)})")
                aligned_actuals = actuals[-len(predictions):]
            else:
                aligned_actuals = actuals

            # This check should now be redundant, but good for safety
            if len(predictions) != len(aligned_actuals):
                logger.warning(f"Skipping weight update for {model_name} due to length mismatch: {len(predictions)} vs {len(aligned_actuals)}")
                continue

            metrics = self.calculate_performance_metrics(predictions, aligned_actuals)
            
            # Use inverse RMSE as score (higher is better)
            if metrics['rmse'] == 0:
                score = 1.0
            elif np.isfinite(metrics['rmse']):
                score = 1.0 / (1.0 + metrics['rmse'])
            else:
                score = 0.0
            
            # Add directional accuracy bonus
            score += metrics['directional_accuracy'] * 0.1
            
            model_scores[model_name] = max(score, 0.01)  # Minimum weight
            
            # Update performance history
            self.model_performance_history[model_name].append(metrics)
            
            # Keep only recent history
            if len(self.model_performance_history[model_name]) > self.performance_window:
                self.model_performance_history[model_name] = \
                    self.model_performance_history[model_name][-self.performance_window:]
        
        # Calculate new weights
        if self.weighting_method == 'performance_based':
            total_score = sum(model_scores.values())
            if total_score > 0:
                self.current_weights = {
                    name: score / total_score 
                    for name, score in model_scores.items()
                }
        
        elif self.weighting_method == 'adaptive':
            # Exponential smoothing with recent performance
            alpha = 0.3  # Smoothing factor
            for model_name, score in model_scores.items():
                current_weight = self.current_weights.get(model_name, 0.0)
                new_weight = alpha * (score / sum(model_scores.values())) + (1 - alpha) * current_weight
                self.current_weights[model_name] = new_weight
        
        logger.info(f"Updated weights: {self.current_weights}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares data for the ensemble model.
        Since child models handle their own data prep, this method primarily ensures
        the target variable is correctly identified.
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")
        
        # The ensemble model itself doesn't do much preparation, as sub-models handle it.
        # We just need to return the dataframe and the target series.
        
        # Ensure date column is present for consistency, though not directly used here
        if self.date_column not in df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in DataFrame.")
            
        target_series = df[self.target_column]
        
        return df, target_series
    
    def fit(self, df: pd.DataFrame) -> 'EnhancedEnsembleModel':
        """Fit all models in the ensemble."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Fitting enhanced ensemble with {len(self.models)} models")
        
        # Fit each model
        fitted_models = []
        for model in self.models:
            try:
                model.fit(df)
                fitted_models.append(model)
                logger.info(f"Fitted {model.name}")
            except Exception as e:
                logger.error(f"Error fitting {model.name}: {e}")
                continue
        
        self.models = fitted_models
        self._initialize_weights()
        
        # Calculate initial performance on training data
        if len(self.models) > 1:
            self._update_weights_on_training_data(df)
        
        self.is_fitted = True
        logger.info("Enhanced ensemble model fitted successfully")
        return self
    
    def _update_weights_on_training_data(self, df: pd.DataFrame):
        """Update weights based on in-sample performance."""
        try:
            # Get in-sample predictions from all models
            model_predictions = {}
            
            for model in self.models:
                try:
                    predictions = model.predict_in_sample(df)
                    if predictions is not None and len(predictions) > 0:
                        model_predictions[model.name] = predictions
                except Exception as e:
                    logger.warning(f"Could not get in-sample predictions from {model.name}: {e}")
                    continue
            
            if len(model_predictions) >= 2:
                # Get actual values
                _, target_series = self.prepare_data(df)
                actuals = target_series.values
                
                # Update weights
                self.update_model_weights(model_predictions, actuals)
                
        except Exception as e:
            logger.warning(f"Could not update weights on training data: {e}")
    
    def predict_with_confidence(self, df: pd.DataFrame, 
                              steps: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        model_predictions = {}
        valid_models = []
        
        for model in self.models:
            try:
                pred = model.predict(df, steps)
                if pred is not None and len(pred) > 0:
                    model_predictions[model.name] = np.array(pred)
                    valid_models.append(model)
            except Exception as e:
                logger.warning(f"Could not get predictions from {model.name}: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("No valid predictions from any model")
        
        # Bootstrap confidence intervals
        ensemble_predictions = []
        
        for _ in range(self.bootstrap_samples):
            # Sample models with replacement based on weights
            sampled_predictions = []
            
            for model_name, predictions in model_predictions.items():
                weight = self.current_weights.get(model_name, 0.0)
                if np.random.random() < weight:
                    # Add noise to create ensemble diversity
                    noise_std = (np.std(predictions) * 0.05) + 1e-8  # 5% noise + epsilon
                    noisy_pred = predictions + np.random.normal(0, noise_std, len(predictions))
                    sampled_predictions.append(noisy_pred)
            
            if sampled_predictions:
                bootstrap_ensemble = np.mean(sampled_predictions, axis=0)
                ensemble_predictions.append(bootstrap_ensemble)
        
        if not ensemble_predictions:
            # Fallback to simple ensemble
            weighted_predictions = []
            total_weight = 0.0
            
            for model_name, predictions in model_predictions.items():
                weight = self.current_weights.get(model_name, 0.0)
                weighted_predictions.append(predictions * weight)
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred = np.sum(weighted_predictions, axis=0) / total_weight
            else:
                ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
            
            return ensemble_pred, None, None
        
        # Calculate confidence intervals
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Central prediction (median)
        central_pred = np.median(ensemble_predictions, axis=0)
        
        # Confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(ensemble_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(ensemble_predictions, upper_percentile, axis=0)
        
        logger.info(f"Generated predictions with {self.confidence_level*100}% confidence intervals")
        
        return central_pred, lower_bound, upper_bound
    
    def predict(self, df: pd.DataFrame, steps: int = 30) -> np.ndarray:
        """Make ensemble predictions (central prediction only)."""
        predictions, _, _ = self.predict_with_confidence(df, steps)
        return predictions
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.current_weights.copy()
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all models."""
        summary = {}
        
        for model_name, history in self.model_performance_history.items():
            if history:
                recent_metrics = history[-min(5, len(history)):]  # Last 5 evaluations
                
                avg_metrics = {}
                for metric in ['rmse', 'mae', 'r2', 'directional_accuracy']:
                    values = [m[metric] for m in recent_metrics if metric in m]
                    if values:
                        avg_metrics[f'avg_{metric}'] = np.mean(values)
                        avg_metrics[f'std_{metric}'] = np.std(values)
                
                summary[model_name] = avg_metrics
        
        return summary
    
    def plot_confidence_bands(self, df: pd.DataFrame, steps: int = 30, 
                            save_path: Optional[str] = None):
        """Plot predictions with confidence bands."""
        
        predictions, lower_bound, upper_bound = self.predict_with_confidence(df, steps)
        
        # Create future dates
        last_date = pd.to_datetime(df[self.date_column]).iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Historical data
        historical_dates = pd.to_datetime(df[self.date_column])
        historical_prices = df[self.target_column]
        
        plt.plot(historical_dates, historical_prices, 
                label='Historical Data', color='black', linewidth=2)
        
        # Predictions
        plt.plot(future_dates, predictions, 
                label='Ensemble Forecast', color='blue', linewidth=2)
        
        # Confidence bands
        plt.fill_between(future_dates, lower_bound, upper_bound, 
                        alpha=0.3, color='blue', 
                        label=f'{self.confidence_level*100}% Confidence Interval')
        
        plt.title('Enhanced Ensemble Forecast with Confidence Intervals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence band plot saved to {save_path}")
        
        plt.show()