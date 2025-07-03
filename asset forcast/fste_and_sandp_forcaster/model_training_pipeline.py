"""
Model Training Pipeline for Forecast Accuracy Assessment.
Orchestrates training and evaluation of multiple forecasting models on a per-ticker basis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import models
from models import (
    ARIMAModel, ProphetModel, LSTMModel, EnsembleModel, ModelEvaluator
)

# Import configuration
from config import PROCESSED_DATA_DIR, MODELS_DIR, TICKER_COLUMN


class ModelTrainingPipeline:
    """
    Comprehensive model training pipeline that trains and evaluates models for each ticker individually.
    """
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data = None
        self.tickers = []
        self.target_column = 'close_price'
        
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
            self.data = pd.read_csv(data_path, low_memory=False)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
            
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} features")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        return self
    
    def prepare_data(self) -> 'ModelTrainingPipeline':
        """
        Prepare data by identifying unique tickers.
        
        Returns:
            ModelTrainingPipeline: Self for chaining
        """
        logger.info("Preparing data for training")
        
        self.data = self.data.dropna(subset=[self.target_column])
        
        if TICKER_COLUMN not in self.data.columns:
            raise ValueError(f"Ticker column '{TICKER_COLUMN}' not found in the dataset.")
            
        self.tickers = self.data[TICKER_COLUMN].unique()
        logger.info(f"Found {len(self.tickers)} tickers. Starting processing for each.")
        
        return self
    
    def run_complete_pipeline(self, test_size: float = 0.2):
        """
        Run the complete training, evaluation, and saving pipeline for each ticker.
        
        Args:
            test_size (float): Proportion of data for testing
            
        Returns:
            Dict[str, Any]: Results for each ticker
        """
        self.load_data()
        self.prepare_data()

        all_results = {}

        for ticker in self.tickers:
            logger.info(f"Processing ticker: {ticker}")
            
            ticker_data = self.data[self.data[TICKER_COLUMN] == ticker].copy()

            if ticker_data['date'].duplicated().any():
                logger.warning(f"Ticker {ticker} has duplicate dates. Dropping them.")
                ticker_data = ticker_data.drop_duplicates(subset=['date'], keep='first')

            if len(ticker_data) < 20:
                logger.warning(f"Skipping ticker {ticker} due to insufficient data (< 20 records).")
                continue

            # 1. Split data
            total_size = len(ticker_data)
            test_samples = int(total_size * test_size)
            train_data = ticker_data.iloc[:-test_samples].copy()
            test_data = ticker_data.iloc[-test_samples:].copy()

            # 2. Initialize models
            arima_model = ARIMAModel(order=(5, 1, 0))
            prophet_model = ProphetModel()
            lstm_model = LSTMModel(sequence_length=30, units=50)
            
            models = {
                'ARIMA': arima_model,
                'Prophet': prophet_model,
                'LSTM': lstm_model
            }

            # 3. Train models
            for model_name, model in models.items():
                try:
                    logger.info(f"Training {model_name} for {ticker}...")
                    
                    model.target_column = self.target_column 
                    
                    train_df = train_data.copy()
                    if model_name in ['Prophet', 'LSTM']:
                        train_df = train_df.rename(columns={'date': 'ds', self.target_column: 'y'})
                        model.target_column = 'y'
                        model.date_column = 'ds'

                    if model_name == 'LSTM':
                        model.fit(train_df, epochs=50)
                    else:
                        model.fit(train_df)

                    logger.info(f"{model_name} for {ticker} trained successfully.")
                except Exception as e:
                    logger.error(f"Error training {model_name} for {ticker}: {e}")
                    models[model_name] = None

            # 4. Evaluate models
            evaluator = ModelEvaluator()
            for model_name, model in models.items():
                if model and model.is_fitted:
                    evaluator.add_model(model, model_name)
            
            ticker_results = evaluator.evaluate_all_models(test_data)
            all_results[ticker] = ticker_results
            logger.info(f"Evaluation results for {ticker}: {ticker_results}")

            # 5. Save models
            for model_name, model in models.items():
                if model and model.is_fitted:
                    model_path = MODELS_DIR / ticker / model_name.lower()
                    model.save_model(str(model_path))

            # 6. Generate reports and visualizations
            report_path = MODELS_DIR / ticker
            report_path.mkdir(parents=True, exist_ok=True)
            evaluator.generate_report(save_path=str(report_path / "evaluation_report.txt"))
            evaluator.plot_comparison('rmse', save_path=str(report_path / "rmse_comparison.png"))
            self.create_visualizations(evaluator, test_data, ticker, report_path)
            
        logger.info("Completed processing all tickers.")
        return all_results
    
    def create_visualizations(self, evaluator, test_data, ticker, save_path):
        """
        Create and save visualizations for a given ticker.
        
        Args:
            evaluator (ModelEvaluator): Evaluator object for generating visualizations
            test_data (pd.DataFrame): Test data for generating predictions
            ticker (str): Ticker identifier
            save_path (Path): Path to save visualizations
        """
        try:
            logger.info(f"Creating prediction plot for {ticker}")
            evaluator.plot_predictions(test_data, save_path=str(save_path / "prediction_plot.png"))

        except Exception as e:
            logger.warning(f"Could not create predictions plot for {ticker}: {e}")


def main():
    """Main function to run the pipeline."""
    logger.add("logs/model_training.log", rotation="500 MB")
    logger.info("============================================================")
    logger.info("STARTING FORECAST MODEL TRAINING PIPELINE")
    logger.info("============================================================")
    
    pipeline = ModelTrainingPipeline()
    try:
        results = pipeline.run_complete_pipeline()
        logger.info("✅ Model training pipeline completed successfully.")
        # Optionally print a summary of results
        for ticker, result in results.items():
            logger.info(f"Results for {ticker}: {result}")
            
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise


if __name__ == "__main__":
    main() 