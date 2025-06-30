"""
Improved Model Training Pipeline with Proper Time Series Validation
Addresses critical issues:
- Proper time-based train/validation/test splits
- Market-specific model training
- Data leakage prevention
- Enhanced evaluation metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import models
from models.improved_lstm_model import ImprovedLSTMModel
from models import ARIMAModel, ProphetModel, EnsembleModel, ModelEvaluator

# Import configuration
from config import PROCESSED_DATA_DIR, MODELS_DIR, TICKER_COLUMN


class ImprovedTrainingPipeline:
    """Enhanced training pipeline with proper time series handling and market-specific training"""
    
    def __init__(self):
        """Initialize the improved training pipeline"""
        self.data = None
        self.tickers = []
        self.target_column = 'close_price'
        self.date_column = 'date'
        
        # Market classification
        self.ftse_tickers = []
        self.sp500_tickers = []
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Improved model training pipeline initialized")
    
    def load_data(self, data_path: Optional[str] = None) -> 'ImprovedTrainingPipeline':
        """Load and prepare data with enhanced preprocessing"""
        if data_path is None:
            data_path = PROCESSED_DATA_DIR / "cleaned_dataset.csv"
        
        logger.info(f"Loading data from {data_path}")
        
        try:
            self.data = pd.read_csv(data_path, low_memory=False)
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data = self.data.sort_values([TICKER_COLUMN, self.date_column])
            
            # Remove any duplicate date-ticker combinations
            self.data = self.data.drop_duplicates(subset=[TICKER_COLUMN, self.date_column], keep='last')
            
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} features")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        return self
    
    def classify_markets(self) -> 'ImprovedTrainingPipeline':
        """Classify tickers by market (FTSE 100 vs S&P 500)"""
        all_tickers = self.data[TICKER_COLUMN].unique()
        
        # FTSE 100 tickers typically end with .L
        self.ftse_tickers = [ticker for ticker in all_tickers if ticker.endswith('.L')]
        self.sp500_tickers = [ticker for ticker in all_tickers if not ticker.endswith('.L')]
        
        logger.info(f"Classified markets - FTSE 100: {len(self.ftse_tickers)}, S&P 500: {len(self.sp500_tickers)}")
        
        return self
    
    def create_time_series_splits(self, ticker_data: pd.DataFrame, 
                                train_ratio: float = 0.7, 
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create proper time-based splits for time series data
        Ensures no data leakage by using chronological order
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Sort by date to ensure chronological order
        ticker_data = ticker_data.sort_values(self.date_column).reset_index(drop=True)
        
        n_samples = len(ticker_data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = ticker_data.iloc[:train_end].copy()
        val_data = ticker_data.iloc[train_end:val_end].copy()
        test_data = ticker_data.iloc[val_end:].copy()
        
        logger.info(f"Time series splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        logger.info(f"Date ranges - Train: {train_data[self.date_column].min()} to {train_data[self.date_column].max()}")
        logger.info(f"             Val: {val_data[self.date_column].min()} to {val_data[self.date_column].max()}")
        logger.info(f"             Test: {test_data[self.date_column].min()} to {test_data[self.date_column].max()}")
        
        return train_data, val_data, test_data
    
    def get_market_specific_config(self, ticker: str) -> Dict[str, Any]:
        """Get market-specific model configurations"""
        if ticker in self.ftse_tickers:
            return {
                'market_type': 'ftse',
                'lstm_config': {
                    'sequence_length': 60,
                    'lstm_units': [64, 32],
                    'dropout_rate': 0.25,
                    'learning_rate': 0.0015
                },
                'arima_config': {
                    'order': (3, 1, 1),  # More conservative for FTSE
                    'seasonal_order': (1, 1, 1, 12)
                },
                'prophet_config': {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0
                }
            }
        else:
            return {
                'market_type': 'sp500',
                'lstm_config': {
                    'sequence_length': 60,
                    'lstm_units': [96, 48],
                    'dropout_rate': 0.35,
                    'learning_rate': 0.001
                },
                'arima_config': {
                    'order': (5, 1, 2),  # More complex for S&P 500
                    'seasonal_order': (1, 1, 1, 12)
                },
                'prophet_config': {
                    'changepoint_prior_scale': 0.1,
                    'seasonality_prior_scale': 15.0
                }
            }
    
    def train_models_for_ticker(self, ticker: str, train_data: pd.DataFrame, 
                              val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models for a specific ticker with market-specific configurations"""
        
        logger.info(f"Training models for {ticker}")
        
        # Get market-specific configuration
        config = self.get_market_specific_config(ticker)
        market_type = config['market_type']
        
        # Initialize models with market-specific configurations
        models = {}
        
        try:
            # Enhanced LSTM Model
            lstm_model = ImprovedLSTMModel(
                market_type=market_type,
                **config['lstm_config']
            )
            
            # Prepare combined training data (train + validation for final training)
            combined_train_data = pd.concat([train_data, val_data]).sort_values(self.date_column)
            
            # Train LSTM
            lstm_model.fit(combined_train_data, target_col=self.target_column, 
                         epochs=50, use_time_series_cv=True)
            models['Enhanced_LSTM'] = lstm_model
            
            logger.info(f"Enhanced LSTM trained successfully for {ticker}")
            
        except Exception as e:
            logger.error(f"Error training Enhanced LSTM for {ticker}: {e}")
            models['Enhanced_LSTM'] = None
        
        try:
            # ARIMA Model with market-specific configuration
            arima_model = ARIMAModel(**config['arima_config'])
            arima_model.target_column = self.target_column
            arima_model.date_column = self.date_column
            
            arima_model.fit(combined_train_data)
            models['ARIMA'] = arima_model
            
            logger.info(f"ARIMA trained successfully for {ticker}")
            
        except Exception as e:
            logger.error(f"Error training ARIMA for {ticker}: {e}")
            models['ARIMA'] = None
        
        try:
            # Prophet Model with market-specific configuration
            prophet_model = ProphetModel(**config['prophet_config'])
            
            # Prepare data for Prophet
            prophet_data = combined_train_data.copy()
            prophet_data = prophet_data.rename(columns={
                self.date_column: 'ds', 
                self.target_column: 'y'
            })
            
            prophet_model.target_column = 'y'
            prophet_model.date_column = 'ds'
            prophet_model.fit(prophet_data)
            models['Prophet'] = prophet_model
            
            logger.info(f"Prophet trained successfully for {ticker}")
            
        except Exception as e:
            logger.error(f"Error training Prophet for {ticker}: {e}")
            models['Prophet'] = None
        
        return models
    
    def evaluate_models_enhanced(self, models: Dict[str, Any], test_data: pd.DataFrame, 
                               ticker: str) -> Dict[str, Any]:
        """Enhanced model evaluation with additional metrics"""
        
        logger.info(f"Evaluating models for {ticker}")
        
        evaluator = ModelEvaluator()
        results = {}
        
        for model_name, model in models.items():
            if model and model.is_fitted:
                try:
                    evaluator.add_model(model, model_name)
                    
                    # Generate predictions
                    if model_name == 'Enhanced_LSTM':
                        predictions = model.predict(test_data, target_col=self.target_column, steps=len(test_data))
                    elif model_name == 'Prophet':
                        # Prepare data for Prophet prediction
                        prophet_test_data = test_data.rename(columns={
                            self.date_column: 'ds',
                            self.target_column: 'y'
                        })
                        predictions = model.predict(prophet_test_data, steps=len(test_data))
                    else:
                        predictions = model.predict(test_data, steps=len(test_data))
                    
                    # Calculate metrics
                    actual_values = test_data[self.target_column].values
                    
                    if len(predictions) == len(actual_values):
                        rmse = np.sqrt(np.mean((predictions - actual_values) ** 2))
                        mae = np.mean(np.abs(predictions - actual_values))
                        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                        
                        # Directional accuracy
                        actual_direction = np.diff(actual_values) > 0
                        pred_direction = np.diff(predictions) > 0
                        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                        
                        # R-squared
                        ss_res = np.sum((actual_values - predictions) ** 2)
                        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        results[model_name] = {
                            'rmse': rmse,
                            'mae': mae,
                            'mape': mape,
                            'r2': r2,
                            'directional_accuracy': directional_accuracy,
                            'predictions': predictions,
                            'actual': actual_values
                        }
                        
                        logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Dir Acc: {directional_accuracy:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} for {ticker}: {e}")
                    results[model_name] = None
        
        return results
    
    def save_models_enhanced(self, models: Dict[str, Any], ticker: str):
        """Save models with enhanced metadata and versioning"""
        ticker_dir = MODELS_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            if model and model.is_fitted:
                try:
                    model_path = ticker_dir / model_name.lower()
                    
                    if model_name == 'Enhanced_LSTM':
                        model.save_model(str(model_path))
                    else:
                        model.save_model(str(model_path))
                    
                    logger.info(f"Saved {model_name} for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error saving {model_name} for {ticker}: {e}")
    
    def generate_enhanced_reports(self, results: Dict[str, Any], ticker: str, 
                                test_data: pd.DataFrame):
        """Generate comprehensive evaluation reports"""
        
        ticker_dir = MODELS_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate summary report
        report_path = ticker_dir / "enhanced_evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Enhanced Model Evaluation Report for {ticker}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Model Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            for model_name, metrics in results.items():
                if metrics:
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
                    f.write(f"  MAE: {metrics['mae']:.4f}\n")
                    f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
                    f.write(f"  R²: {metrics['r2']:.4f}\n")
                    f.write(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%\n")
            
            # Market-specific insights
            market_type = "FTSE 100" if ticker.endswith('.L') else "S&P 500"
            f.write(f"\nMarket Type: {market_type}\n")
            f.write(f"Test Period: {test_data[self.date_column].min()} to {test_data[self.date_column].max()}\n")
            f.write(f"Number of Test Samples: {len(test_data)}\n")
        
        logger.info(f"Enhanced evaluation report saved for {ticker}")
    
    def run_improved_pipeline(self, 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15, 
                            test_ratio: float = 0.15) -> Dict[str, Any]:
        """
        Run the complete improved training pipeline
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation  
            test_ratio: Proportion of data for testing
            
        Returns:
            Dict containing results for each ticker
        """
        logger.info("Starting improved model training pipeline")
        
        # Load and prepare data
        self.load_data()
        self.classify_markets()
        
        if TICKER_COLUMN not in self.data.columns:
            raise ValueError(f"Ticker column '{TICKER_COLUMN}' not found in dataset")
        
        self.tickers = self.data[TICKER_COLUMN].unique()
        logger.info(f"Processing {len(self.tickers)} tickers")
        
        all_results = {}
        
        for ticker in self.tickers:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing ticker: {ticker}")
            logger.info(f"{'='*50}")
            
            # Get ticker data
            ticker_data = self.data[self.data[TICKER_COLUMN] == ticker].copy()
            
            # Skip if insufficient data
            if len(ticker_data) < 100:
                logger.warning(f"Skipping {ticker} - insufficient data ({len(ticker_data)} records)")
                continue
            
            # Create proper time series splits
            try:
                train_data, val_data, test_data = self.create_time_series_splits(
                    ticker_data, train_ratio, val_ratio, test_ratio
                )
                
                # Skip if any split is too small
                if len(train_data) < 50 or len(test_data) < 10:
                    logger.warning(f"Skipping {ticker} - splits too small")
                    continue
                
                # Train models
                models = self.train_models_for_ticker(ticker, train_data, val_data)
                
                # Evaluate models
                results = self.evaluate_models_enhanced(models, test_data, ticker)
                
                # Save models
                self.save_models_enhanced(models, ticker)
                
                # Generate reports
                self.generate_enhanced_reports(results, ticker, test_data)
                
                all_results[ticker] = {
                    'models': models,
                    'results': results,
                    'data_info': {
                        'train_samples': len(train_data),
                        'val_samples': len(val_data),
                        'test_samples': len(test_data),
                        'train_period': f"{train_data[self.date_column].min()} to {train_data[self.date_column].max()}",
                        'test_period': f"{test_data[self.date_column].min()} to {test_data[self.date_column].max()}"
                    }
                }
                
                logger.info(f"Successfully processed {ticker}")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                all_results[ticker] = {'error': str(e)}
        
        logger.info(f"\nPipeline completed. Processed {len(all_results)} tickers.")
        
        # Generate overall summary
        self._generate_overall_summary(all_results)
        
        return all_results
    
    def _generate_overall_summary(self, all_results: Dict[str, Any]):
        """Generate overall pipeline summary"""
        summary_path = MODELS_DIR / "pipeline_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Improved Training Pipeline Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            successful_tickers = [ticker for ticker, result in all_results.items() 
                                if 'error' not in result]
            failed_tickers = [ticker for ticker, result in all_results.items() 
                            if 'error' in result]
            
            f.write(f"Total tickers processed: {len(all_results)}\n")
            f.write(f"Successful: {len(successful_tickers)}\n")
            f.write(f"Failed: {len(failed_tickers)}\n\n")
            
            if failed_tickers:
                f.write("Failed tickers:\n")
                for ticker in failed_tickers:
                    f.write(f"  - {ticker}: {all_results[ticker]['error']}\n")
                f.write("\n")
            
            # Market-wise performance summary
            ftse_results = []
            sp500_results = []
            
            for ticker, result in all_results.items():
                if 'results' in result:
                    if ticker.endswith('.L'):
                        ftse_results.append(result['results'])
                    else:
                        sp500_results.append(result['results'])
            
            f.write("Market Performance Summary:\n")
            f.write(f"FTSE 100 stocks: {len(ftse_results)}\n")
            f.write(f"S&P 500 stocks: {len(sp500_results)}\n")
        
        logger.info(f"Overall pipeline summary saved to {summary_path}")


def main():
    """Main function to run the improved pipeline"""
    logger.add("logs/improved_training.log", rotation="500 MB")
    
    pipeline = ImprovedTrainingPipeline()
    results = pipeline.run_improved_pipeline()
    
    logger.info("Improved training pipeline completed successfully!")


if __name__ == "__main__":
    main() 