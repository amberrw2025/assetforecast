"""
Model Training Pipeline for Forecast Accuracy Assessment.
Orchestrates training and evaluation of multiple forecasting models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import models
from models import (
    ARIMAModel, ProphetModel, LSTMModel, EnsembleModel, ModelEvaluator
)

# Import configuration
from config import PROCESSED_DATA_DIR, MODELS_DIR


class ModelTrainingPipeline:
    """
    Comprehensive model training pipeline for forecast accuracy assessment.
    """
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.evaluator = ModelEvaluator()
        self.results = {}
        
        # Create models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model training pipeline initialized")
    
    def load_data(self, data_path: Optional[str] = None) -> 'ModelTrainingPipeline':
        """
        Load and prepare data for training.
        
        Args:
            data_path (Optional[str]): Path to the cleaned dataset
            
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        if data_path is None:
            data_path = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
        
        logger.info(f"Loading data from {data_path}")
        
        try:
            self.data = pd.read_csv(data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
            
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} features")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        return self
    
    def prepare_data(self, test_size: float = 0.2, 
                    target_column: str = 'close_price') -> 'ModelTrainingPipeline':
        """
        Prepare data for training and testing.
        
        Args:
            test_size (float): Proportion of data for testing
            target_column (str): Target variable column
            
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        logger.info("Preparing data for training")
        
        # Remove rows with missing target values
        self.data = self.data.dropna(subset=[target_column])
        
        # Split data by date
        total_size = len(self.data)
        test_samples = int(total_size * test_size)
        
        self.train_data = self.data.iloc[:-test_samples].copy()
        self.test_data = self.data.iloc[-test_samples:].copy()
        
        logger.info(f"Training set: {len(self.train_data)} records")
        logger.info(f"Test set: {len(self.test_data)} records")
        logger.info(f"Date range - Train: {self.train_data['date'].min()} to {self.train_data['date'].max()}")
        logger.info(f"Date range - Test: {self.test_data['date'].min()} to {self.test_data['date'].max()}")
        
        return self
    
    def initialize_models(self) -> 'ModelTrainingPipeline':
        """
        Initialize all forecasting models.
        
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        logger.info("Initializing forecasting models")
        
        # ARIMA Model
        arima_model = ARIMAModel(order=(1, 1, 1))
        self.models['ARIMA'] = arima_model
        self.evaluator.add_model(arima_model, 'ARIMA')
        
        # Prophet Model
        prophet_model = ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        self.models['Prophet'] = prophet_model
        self.evaluator.add_model(prophet_model, 'Prophet')
        
        # LSTM Model
        lstm_model = LSTMModel(
            sequence_length=30,
            units=50,
            layers=2,
            dropout=0.2,
            learning_rate=0.001
        )
        self.models['LSTM'] = lstm_model
        self.evaluator.add_model(lstm_model, 'LSTM')
        
        # Ensemble Model
        ensemble_model = EnsembleModel(
            models=[arima_model, prophet_model, lstm_model],
            weights=[0.3, 0.3, 0.4],
            ensemble_method='weighted_average'
        )
        self.models['Ensemble'] = ensemble_model
        self.evaluator.add_model(ensemble_model, 'Ensemble')
        
        logger.info(f"Initialized {len(self.models)} models")
        return self
    
    def train_models(self) -> 'ModelTrainingPipeline':
        """
        Train all models.
        
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        logger.info("Starting model training")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                model.fit(self.train_data)
                logger.info(f"{model_name} training completed")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return self
    
    def evaluate_models(self) -> 'ModelTrainingPipeline':
        """
        Evaluate all trained models.
        
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        logger.info("Evaluating models")
        
        self.results = self.evaluator.evaluate_all_models(self.test_data)
        
        return self
    
    def generate_forecasts(self, steps: int = 30) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for all models.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Dict[str, np.ndarray]: Forecasts for each model
        """
        logger.info(f"Generating {steps}-step forecasts")
        
        forecasts = {}
        
        for model_name, model in self.models.items():
            try:
                forecast = model.predict(self.test_data, steps)
                forecasts[model_name] = forecast
                logger.info(f"Generated forecast for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {e}")
                continue
        
        return forecasts
    
    def save_models(self) -> 'ModelTrainingPipeline':
        """
        Save all trained models.
        
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        logger.info("Saving trained models")
        
        for model_name, model in self.models.items():
            try:
                if model.is_fitted:
                    model_path = MODELS_DIR / model_name.lower()
                    model.save_model(str(model_path))
                    logger.info(f"Saved {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
                continue
        
        return self
    
    def create_visualizations(self, save_dir: Optional[str] = None) -> 'ModelTrainingPipeline':
        """
        Create comprehensive visualizations.
        
        Args:
            save_dir (Optional[str]): Directory to save visualizations
            
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        if save_dir is None:
            save_dir = PROCESSED_DATA_DIR
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating visualizations")
        
        # Model comparison plot
        try:
            self.evaluator.plot_comparison('rmse', 
                                         save_path / 'model_comparison_rmse.png')
        except Exception as e:
            logger.warning(f"Could not create RMSE comparison plot: {e}")
        
        # Predictions plot
        try:
            self.evaluator.plot_predictions(self.test_data, 
                                          save_path / 'model_predictions.png')
        except Exception as e:
            logger.warning(f"Could not create predictions plot: {e}")
        
        # LSTM training history (if available)
        if 'LSTM' in self.models and self.models['LSTM'].is_fitted:
            try:
                self.models['LSTM'].plot_training_history(
                    save_path / 'lstm_training_history.png'
                )
            except Exception as e:
                logger.warning(f"Could not create LSTM training history plot: {e}")
        
        # ARIMA diagnostics (if available)
        if 'ARIMA' in self.models and self.models['ARIMA'].is_fitted:
            try:
                self.models['ARIMA'].plot_diagnostics(
                    save_path / 'arima_diagnostics.png'
                )
            except Exception as e:
                logger.warning(f"Could not create ARIMA diagnostics plot: {e}")
        
        return self
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            save_path (Optional[str]): Path to save the report
            
        Returns:
            str: Report content
        """
        if save_path is None:
            save_path = PROCESSED_DATA_DIR / "model_training_report.txt"
        
        report = self.evaluator.generate_report(save_path)
        
        # Add pipeline-specific information
        additional_info = f"""
PIPELINE INFORMATION
===================
Training Records: {len(self.train_data)}
Test Records: {len(self.test_data)}
Models Trained: {len([m for m in self.models.values() if m.is_fitted])}
Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA INFORMATION
===============
Target Variable: close_price
Date Range: {self.data['date'].min()} to {self.data['date'].max()}
Total Features: {len(self.data.columns)}
Missing Values: {self.data.isnull().sum().sum()}

MODEL CONFIGURATIONS
===================
"""
        
        for model_name, model in self.models.items():
            additional_info += f"\n{model_name}:\n"
            additional_info += f"  Parameters: {model.model_params}\n"
            additional_info += f"  Fitted: {model.is_fitted}\n"
        
        # Append to report
        with open(save_path, 'a') as f:
            f.write(additional_info)
        
        logger.info(f"Training report saved to {save_path}")
        return report
    
    def run_complete_pipeline(self, 
                            data_path: Optional[str] = None,
                            test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run the complete model training pipeline.
        
        Args:
            data_path (Optional[str]): Path to the cleaned dataset
            test_size (float): Proportion of data for testing
            
        Returns:
            Dict[str, Any]: Pipeline results
        """
        logger.info("Starting complete model training pipeline")
        
        try:
            # Execute pipeline steps
            (self.load_data(data_path)
             .prepare_data(test_size)
             .initialize_models()
             .train_models()
             .evaluate_models()
             .save_models()
             .create_visualizations()
             .generate_report())
            
            # Generate forecasts
            forecasts = self.generate_forecasts()
            
            # Get best model
            best_model, best_score = self.evaluator.get_best_model('rmse')
            
            results = {
                'models_trained': len([m for m in self.models.values() if m.is_fitted]),
                'evaluation_results': self.results,
                'best_model': best_model,
                'best_score': best_score,
                'forecasts': forecasts,
                'training_data_size': len(self.train_data),
                'test_data_size': len(self.test_data)
            }
            
            logger.info("Model training pipeline completed successfully")
            logger.info(f"Best model: {best_model} (RMSE: {best_score:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise


def main():
    """Main function to run the model training pipeline."""
    print("=" * 60)
    print("FORECAST MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"📊 Models trained: {results['models_trained']}")
        print(f"🏆 Best model: {results['best_model']}")
        print(f"📈 Best RMSE: {results['best_score']:.4f}")
        print(f"📁 Models saved to: {MODELS_DIR}")
        print(f"📊 Results saved to: {PROCESSED_DATA_DIR}")
        print("\n🚀 Ready for deployment and production use!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        print("Check the logs for detailed error information.")
        raise


if __name__ == "__main__":
    main() 