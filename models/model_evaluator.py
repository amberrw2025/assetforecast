"""
Model evaluator for comprehensive model assessment and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from .base_model import BaseForecastModel


class ModelEvaluator:
    """
    Comprehensive model evaluator for forecast accuracy assessment.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.models = {}
        self.results = {}
        self.comparison_results = {}
    
    def add_model(self, model: BaseForecastModel, name: Optional[str] = None) -> 'ModelEvaluator':
        """
        Add a model for evaluation.
        
        Args:
            model (BaseForecastModel): Model to evaluate
            name (Optional[str]): Custom name for the model
            
        Returns:
            ModelEvaluator: Self for chaining
        """
        model_name = name or model.name
        self.models[model_name] = model
        logger.info(f"Added model: {model_name}")
        return self
    
    def evaluate_model(self, model_name: str, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model_name (str): Name of the model to evaluate
            df (pd.DataFrame): Test data
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            metrics = model.evaluate(df)
            self.results[model_name] = metrics
            logger.info(f"Evaluated {model_name}: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {}
    
    def evaluate_all_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models.
        
        Args:
            df (pd.DataFrame): Test data
            
        Returns:
            Dict[str, Dict[str, float]]: All model results
        """
        logger.info("Evaluating all models")
        
        for model_name in self.models:
            self.evaluate_model(model_name, df)
        
        return self.results
    
    def compare_models(self, metric: str = 'rmse') -> pd.DataFrame:
        """
        Compare models based on a specific metric.
        
        Args:
            metric (str): Metric to compare on
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            if metric in metrics:
                comparison_data.append({
                    'Model': model_name,
                    metric.upper(): metrics[metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric.upper())
        
        self.comparison_results[metric] = comparison_df
        
        logger.info(f"Model comparison by {metric}:")
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_comparison(self, metric: str = 'rmse', save_path: Optional[str] = None) -> None:
        """
        Plot model comparison.
        
        Args:
            metric (str): Metric to plot
            save_path (Optional[str]): Path to save the plot
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        # Prepare data
        model_names = []
        metric_values = []
        
        for model_name, metrics in self.results.items():
            if metric in metrics:
                model_names.append(model_name)
                metric_values.append(metrics[metric])
        
        if not model_names:
            raise ValueError(f"No data available for metric: {metric}")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        bars = plt.bar(model_names, metric_values)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, df: pd.DataFrame, 
                        models: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Plot predictions from all models.
        
        Args:
            df (pd.DataFrame): Test data
            models (Optional[List[str]]): Models to plot
            save_path (Optional[str]): Path to save the plot
        """
        if not models:
            models = list(self.models.keys())
        
        plt.figure(figsize=(15, 8))
        
        # Get true values
        true_values = df['close_price'].dropna()
        dates = df['date'].dropna()
        
        plt.plot(dates, true_values, label='True Values', linewidth=2, color='black')
        
        # Plot predictions for each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, model_name in enumerate(models):
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            
            try:
                predictions = model.predict_in_sample(df)
                
                # Remove NaN values
                mask = ~np.isnan(predictions)
                pred_dates = dates[mask]
                pred_values = predictions[mask]
                
                plt.plot(pred_dates, pred_values, 
                        label=f'{model_name} Predictions', 
                        color=colors[i], alpha=0.7)
                
            except Exception as e:
                logger.warning(f"Could not plot predictions for {model_name}: {e}")
        
        plt.title('Model Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path (Optional[str]): Path to save the report
            
        Returns:
            str: Report content
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        report = []
        report.append("=" * 60)
        report.append("FORECAST MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models evaluated: {len(self.models)}")
        report.append("")
        
        # Summary table
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        summary_data = []
        
        for model_name, model_metrics in self.results.items():
            row = [model_name]
            for metric in metrics:
                value = model_metrics.get(metric, np.nan)
                row.append(f"{value:.4f}" if not np.isnan(value) else "N/A")
            summary_data.append(row)
        
        # Create summary table
        headers = ['Model'] + [metric.upper() for metric in metrics]
        summary_df = pd.DataFrame(summary_data, columns=headers)
        report.append(summary_df.to_string(index=False))
        report.append("")
        
        # Best model by each metric
        report.append("BEST MODELS BY METRIC")
        report.append("-" * 30)
        
        for metric in metrics:
            best_model = None
            best_value = np.inf if metric != 'r2' else -np.inf
            
            for model_name, model_metrics in self.results.items():
                if metric in model_metrics:
                    value = model_metrics[metric]
                    if not np.isnan(value):
                        if metric == 'r2':
                            if value > best_value:
                                best_value = value
                                best_model = model_name
                        else:
                            if value < best_value:
                                best_value = value
                                best_model = model_name
            
            if best_model:
                report.append(f"{metric.upper()}: {best_model} ({best_value:.4f})")
        
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 20)
        
        for model_name, metrics in self.results.items():
            report.append(f"\n{model_name}:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    report.append(f"  {metric.upper()}: {value:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        report_content = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_content
    
    def cross_validation(self, df: pd.DataFrame, 
                        n_splits: int = 5,
                        test_size: float = 0.2) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Args:
            df (pd.DataFrame): Full dataset
            n_splits (int): Number of splits
            test_size (float): Test set size ratio
            
        Returns:
            Dict[str, List[float]]: Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation")
        
        # Prepare data
        df = df.sort_values('date')
        total_size = len(df)
        test_samples = int(total_size * test_size)
        
        cv_results = {model_name: [] for model_name in self.models}
        
        for i in range(n_splits):
            # Calculate split indices
            start_idx = i * test_samples
            end_idx = start_idx + test_samples
            
            # Split data
            train_df = df.iloc[:start_idx]
            test_df = df.iloc[start_idx:end_idx]
            
            if len(train_df) < 100 or len(test_df) < 10:
                continue
            
            logger.info(f"CV fold {i+1}/{n_splits}: train={len(train_df)}, test={len(test_df)}")
            
            # Fit and evaluate each model
            for model_name, model in self.models.items():
                try:
                    # Create a copy of the model for this fold
                    model_copy = type(model)(**model.model_params)
                    model_copy.fit(train_df)
                    
                    # Evaluate
                    metrics = model_copy.evaluate(test_df)
                    cv_results[model_name].append(metrics.get('rmse', np.nan))
                    
                except Exception as e:
                    logger.warning(f"Error in CV fold {i+1} for {model_name}: {e}")
                    cv_results[model_name].append(np.nan)
        
        # Calculate CV statistics
        cv_summary = {}
        for model_name, results in cv_results.items():
            valid_results = [r for r in results if not np.isnan(r)]
            if valid_results:
                cv_summary[model_name] = {
                    'mean_rmse': np.mean(valid_results),
                    'std_rmse': np.std(valid_results),
                    'n_folds': len(valid_results)
                }
        
        logger.info("Cross-validation completed")
        return cv_summary
    
    def get_best_model(self, metric: str = 'rmse') -> Tuple[str, float]:
        """
        Get the best performing model.
        
        Args:
            metric (str): Metric to optimize for
            
        Returns:
            Tuple[str, float]: Best model name and score
        """
        if not self.results:
            raise ValueError("No evaluation results available")
        
        best_model = None
        best_score = np.inf if metric != 'r2' else -np.inf
        
        for model_name, metrics in self.results.items():
            if metric in metrics:
                score = metrics[metric]
                if not np.isnan(score):
                    if metric == 'r2':
                        if score > best_score:
                            best_score = score
                            best_model = model_name
                    else:
                        if score < best_score:
                            best_score = score
                            best_model = model_name
        
        if best_model is None:
            raise ValueError(f"No valid results for metric: {metric}")
        
        return best_model, best_score 