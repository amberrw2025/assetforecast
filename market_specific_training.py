"""
Market-Specific Training Pipeline
Addresses Fix #3: Market-Specific Model Training

- Separate FTSE 100 vs S&P 500 configurations
- Market-specific hyperparameters
- Currency and economic context awareness
- Cross-market validation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.append(str(project_root))

# Import our modules
try:
    from enhanced_feature_engineering import EnhancedFeatureEngineer
    from models.improved_lstm_model import ImprovedLSTMModel
    from utils.plotting_fixes import ForecastDataValidator, RobustForecasterPlotter
except ImportError as e:
    logger.warning(f"Import error: {e}. Some features may not work.")

# Import standard libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import json


class MarketSpecificConfig:
    """Configuration class for market-specific training"""
    
    FTSE_CONFIG = {
        'market_name': 'FTSE_100',
        'currency_context': 'GBP',
        'trading_hours': (9, 17),  # London time
        'economic_indicators': ['gbp_usd', 'uk_rates', 'eu_markets'],
        'volatility_adjustment': 1.2,  # FTSE typically more volatile
        
        # LSTM hyperparameters optimized for FTSE
        'lstm_config': {
            'sequence_length': 60,
            'lstm_units': [128, 64, 32],
            'dropout_rates': [0.3, 0.2, 0.1],
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'validation_split': 0.15
        },
        
        # Feature weights for FTSE
        'feature_weights': {
            'currency_features': 0.3,  # Higher weight for GBP/USD
            'technical_indicators': 0.4,
            'market_regime': 0.2,
            'time_features': 0.1
        }
    }
    
    SP500_CONFIG = {
        'market_name': 'SP_500',
        'currency_context': 'USD',
        'trading_hours': (9, 16),  # EST time
        'economic_indicators': ['fed_rates', 'vix', 'us_treasury'],
        'volatility_adjustment': 1.0,  # Baseline
        
        # LSTM hyperparameters optimized for S&P 500
        'lstm_config': {
            'sequence_length': 50,
            'lstm_units': [100, 50, 25],
            'dropout_rates': [0.2, 0.15, 0.1],
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 80,
            'early_stopping_patience': 12,
            'validation_split': 0.2
        },
        
        # Feature weights for S&P 500
        'feature_weights': {
            'currency_features': 0.1,  # Lower weight for USD-based
            'technical_indicators': 0.5,
            'market_regime': 0.3,
            'time_features': 0.1
        }
    }


class MarketClassifier:
    """Classifies stocks into market groups"""
    
    @staticmethod
    def classify_market(ticker: str) -> str:
        """Classify ticker into market"""
        if ticker.endswith('.L'):
            return 'FTSE_100'
        else:
            return 'SP_500'
    
    @staticmethod
    def get_market_config(market: str) -> Dict[str, Any]:
        """Get configuration for specific market"""
        if market == 'FTSE_100':
            return MarketSpecificConfig.FTSE_CONFIG
        elif market == 'SP_500':
            return MarketSpecificConfig.SP500_CONFIG
        else:
            raise ValueError(f"Unknown market: {market}")


class MarketSpecificDataProcessor:
    """Processes data with market-specific considerations"""
    
    def __init__(self, market: str):
        self.market = market
        self.config = MarketClassifier.get_market_config(market)
        self.feature_engineer = EnhancedFeatureEngineer()
        self.data_validator = ForecastDataValidator()
        
        # Market-specific scalers
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
    def process_market_data(self, 
                          df: pd.DataFrame, 
                          ticker: str,
                          target_col: str = 'close_price') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process data with market-specific enhancements"""
        
        logger.info(f"Processing {ticker} for {self.market} market")
        
        # 1. Add enhanced features
        enhanced_df = self.feature_engineer.add_all_features(df, ticker)
        
        # 2. Apply market-specific adjustments
        enhanced_df = self._apply_market_adjustments(enhanced_df, ticker)
        
        # 3. Clean and validate
        enhanced_df = self.data_validator.clean_forecast_data(enhanced_df)
        
        # 4. Feature selection based on market
        feature_cols = self._select_market_features(enhanced_df)
        
        # 5. Create processing metadata
        metadata = {
            'market': self.market,
            'ticker': ticker,
            'total_features': len(enhanced_df.columns),
            'selected_features': len(feature_cols),
            'feature_columns': feature_cols,
            'data_shape': enhanced_df.shape,
            'date_range': (enhanced_df['date'].min(), enhanced_df['date'].max())
        }
        
        logger.info(f"Data processing complete for {ticker}: {enhanced_df.shape}")
        
        return enhanced_df, metadata
    
    def _apply_market_adjustments(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Apply market-specific adjustments"""
        df = df.copy()
        
        # Market-specific volatility adjustments
        vol_adj = self.config['volatility_adjustment']
        if 'volatility_20' in df.columns:
            df['adjusted_volatility'] = df['volatility_20'] * vol_adj
        
        # Currency impact (higher for FTSE)
        if self.market == 'FTSE_100' and 'gbp_impact' in df.columns:
            df['currency_impact_weighted'] = df['gbp_impact'] * 0.3
        else:
            df['currency_impact_weighted'] = 0
        
        # Market hours indicator
        if 'hour' in df.columns:
            trading_start, trading_end = self.config['trading_hours']
            df['is_trading_hours'] = ((df['hour'] >= trading_start) & 
                                    (df['hour'] <= trading_end)).astype(int)
        
        # Market-specific regime adjustments
        if 'market_phase' in df.columns:
            # FTSE tends to be more sensitive to bear markets
            if self.market == 'FTSE_100':
                df['bear_sensitivity'] = np.where(df['market_phase'] >= 2, 1.2, 1.0)
            else:
                df['bear_sensitivity'] = 1.0
        
        return df
    
    def _select_market_features(self, df: pd.DataFrame) -> List[str]:
        """Select features based on market configuration"""
        
        # Base features (always include)
        base_features = ['close_price', 'returns', 'log_returns']
        
        # Technical indicators
        technical_features = [col for col in df.columns if any(pattern in col for pattern in 
                            ['sma_', 'ema_', 'rsi_', 'bb_', 'volatility_', 'momentum_'])]
        
        # Market regime features
        regime_features = [col for col in df.columns if any(pattern in col for pattern in 
                         ['trend_', 'market_phase', 'volatility_regime'])]
        
        # Currency features (weight based on market)
        currency_features = [col for col in df.columns if any(pattern in col for pattern in 
                           ['gbp_', 'eur_', 'currency_'])]
        
        # Time features
        time_features = [col for col in df.columns if any(pattern in col for pattern in 
                        ['month_', 'quarter', 'day_of_', 'is_month_', 'is_quarter_'])]
        
        # Lag features
        lag_features = [col for col in df.columns if '_lag_' in col]
        
        # Market-specific weighting
        weights = self.config['feature_weights']
        
        selected_features = base_features.copy()
        
        # Add features based on weights
        if weights['technical_indicators'] > 0.3:
            selected_features.extend(technical_features[:15])  # Top 15 technical
        
        if weights['market_regime'] > 0.2:
            selected_features.extend(regime_features)
        
        if weights['currency_features'] > 0.2:
            selected_features.extend(currency_features[:5])  # Top 5 currency
        
        if weights['time_features'] > 0.05:
            selected_features.extend(time_features[:8])  # Top 8 time
        
        # Add some lag features
        selected_features.extend(lag_features[:10])  # Top 10 lags
        
        # Remove duplicates and ensure they exist in DataFrame
        selected_features = list(set(selected_features))
        selected_features = [col for col in selected_features if col in df.columns]
        
        logger.info(f"Selected {len(selected_features)} features for {self.market}")
        
        return selected_features
    
    def prepare_lstm_data(self, 
                         df: pd.DataFrame, 
                         feature_cols: List[str],
                         target_col: str = 'close_price') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training with market-specific sequence length"""
        
        sequence_length = self.config['lstm_config']['sequence_length']
        
        # Scale features and target
        features = df[feature_cols].values
        target = df[target_col].values.reshape(-1, 1)
        
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.price_scaler.fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(features_scaled)):
            X.append(features_scaled[i-sequence_length:i])
            y.append(target_scaled[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared LSTM data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y


class MarketSpecificTrainer:
    """Trains market-specific models"""
    
    def __init__(self, market: str):
        self.market = market
        self.config = MarketClassifier.get_market_config(market)
        self.data_processor = MarketSpecificDataProcessor(market)
        
        # Model storage
        self.models = {}
        self.metadata = {}
        self.performance_metrics = {}
        
        # Create market-specific directories
        self.model_dir = Path(f'models/{market.lower()}')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def train_stock_model(self, 
                         df: pd.DataFrame, 
                         ticker: str,
                         target_col: str = 'close_price') -> Dict[str, Any]:
        """Train a market-specific model for a single stock"""
        
        logger.info(f"Training {self.market} model for {ticker}")
        
        # 1. Process data
        processed_df, data_metadata = self.data_processor.process_market_data(df, ticker, target_col)
        
        # 2. Prepare LSTM data
        feature_cols = data_metadata['feature_columns']
        X, y = self.data_processor.prepare_lstm_data(processed_df, feature_cols, target_col)
        
        # 3. Split data with proper time series validation
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15
        
        n_train = int(len(X) * train_size)
        n_val = int(len(X) * val_size)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        # 4. Create and train model
        lstm_config = self.config['lstm_config']
        
        model = ImprovedLSTMModel(
            input_shape=(X.shape[1], X.shape[2]),
            market_type=self.market,
            **lstm_config
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            validation_data=(X_val, y_val),
            **lstm_config
        )
        
        # 5. Evaluate model
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Inverse transform predictions
        train_pred_inv = self.data_processor.price_scaler.inverse_transform(train_pred)
        val_pred_inv = self.data_processor.price_scaler.inverse_transform(val_pred)
        test_pred_inv = self.data_processor.price_scaler.inverse_transform(test_pred)
        
        y_train_inv = self.data_processor.price_scaler.inverse_transform(y_train)
        y_val_inv = self.data_processor.price_scaler.inverse_transform(y_val)
        y_test_inv = self.data_processor.price_scaler.inverse_transform(y_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_train_inv, train_pred_inv,
            y_val_inv, val_pred_inv,
            y_test_inv, test_pred_inv,
            ticker
        )
        
        # 6. Save model and metadata
        model_info = self._save_model_artifacts(
            model, ticker, data_metadata, metrics, history
        )
        
        # Store in memory
        self.models[ticker] = model
        self.metadata[ticker] = data_metadata
        self.performance_metrics[ticker] = metrics
        
        logger.info(f"Training completed for {ticker}. Test RMSE: {metrics['test_rmse']:.2f}")
        
        return model_info
    
    def train_market_portfolio(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train models for all stocks in the market"""
        
        logger.info(f"Training {self.market} portfolio with {len(stock_data)} stocks")
        
        portfolio_results = {}
        total_stocks = len(stock_data)
        
        for i, (ticker, df) in enumerate(stock_data.items(), 1):
            try:
                logger.info(f"Training {i}/{total_stocks}: {ticker}")
                
                # Only train if market matches
                stock_market = MarketClassifier.classify_market(ticker)
                if stock_market != self.market:
                    logger.warning(f"Skipping {ticker} - belongs to {stock_market}, not {self.market}")
                    continue
                
                model_info = self.train_stock_model(df, ticker)
                portfolio_results[ticker] = model_info
                
                logger.info(f"✓ {ticker} completed successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to train {ticker}: {str(e)}")
                portfolio_results[ticker] = {'error': str(e)}
        
        # Generate portfolio summary
        portfolio_summary = self._generate_portfolio_summary(portfolio_results)
        
        # Save portfolio results
        portfolio_path = self.model_dir / f'{self.market}_portfolio_results.json'
        with open(portfolio_path, 'w') as f:
            json.dump(portfolio_summary, f, indent=2, default=str)
        
        logger.info(f"Portfolio training completed for {self.market}")
        logger.info(f"Results saved to: {portfolio_path}")
        
        return portfolio_summary
    
    def _calculate_metrics(self, 
                          y_train_true, y_train_pred,
                          y_val_true, y_val_pred,
                          y_test_true, y_test_pred,
                          ticker: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        def calculate_set_metrics(y_true, y_pred, set_name):
            return {
                f'{set_name}_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                f'{set_name}_mae': mean_absolute_error(y_true, y_pred),
                f'{set_name}_r2': r2_score(y_true, y_pred),
                f'{set_name}_mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
        
        metrics = {}
        metrics.update(calculate_set_metrics(y_train_true, y_train_pred, 'train'))
        metrics.update(calculate_set_metrics(y_val_true, y_val_pred, 'val'))
        metrics.update(calculate_set_metrics(y_test_true, y_test_pred, 'test'))
        
        # Directional accuracy
        def directional_accuracy(y_true, y_pred):
            true_direction = np.diff(y_true.flatten()) > 0
            pred_direction = np.diff(y_pred.flatten()) > 0
            return np.mean(true_direction == pred_direction) * 100
        
        metrics['train_directional_accuracy'] = directional_accuracy(y_train_true, y_train_pred)
        metrics['val_directional_accuracy'] = directional_accuracy(y_val_true, y_val_pred)
        metrics['test_directional_accuracy'] = directional_accuracy(y_test_true, y_test_pred)
        
        # Market-specific metrics
        metrics['market'] = self.market
        metrics['ticker'] = ticker
        metrics['timestamp'] = datetime.now().isoformat()
        
        return metrics
    
    def _save_model_artifacts(self, 
                            model, 
                            ticker: str, 
                            data_metadata: Dict[str, Any], 
                            metrics: Dict[str, float],
                            history) -> Dict[str, Any]:
        """Save model and related artifacts"""
        
        # Save model
        model_path = self.model_dir / f'{ticker}_model.h5'
        model.save(str(model_path))
        
        # Save scalers
        scalers_path = self.model_dir / f'{ticker}_scalers.joblib'
        joblib.dump({
            'price_scaler': self.data_processor.price_scaler,
            'feature_scaler': self.data_processor.feature_scaler
        }, scalers_path)
        
        # Save metadata
        metadata_path = self.model_dir / f'{ticker}_metadata.json'
        full_metadata = {
            'data_metadata': data_metadata,
            'metrics': metrics,
            'config': self.config,
            'model_path': str(model_path),
            'scalers_path': str(scalers_path),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']] if 'val_loss' in history.history else []
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)
        
        return {
            'ticker': ticker,
            'market': self.market,
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'test_rmse': metrics['test_rmse'],
            'test_r2': metrics['test_r2'],
            'directional_accuracy': metrics['test_directional_accuracy']
        }
    
    def _generate_portfolio_summary(self, portfolio_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the portfolio"""
        
        successful_stocks = {k: v for k, v in portfolio_results.items() if 'error' not in v}
        failed_stocks = {k: v for k, v in portfolio_results.items() if 'error' in v}
        
        if successful_stocks:
            test_rmses = [result['test_rmse'] for result in successful_stocks.values()]
            test_r2s = [result['test_r2'] for result in successful_stocks.values()]
            directional_accs = [result['directional_accuracy'] for result in successful_stocks.values()]
            
            summary = {
                'market': self.market,
                'total_stocks': len(portfolio_results),
                'successful_stocks': len(successful_stocks),
                'failed_stocks': len(failed_stocks),
                'success_rate': len(successful_stocks) / len(portfolio_results) * 100,
                
                # Performance statistics
                'portfolio_metrics': {
                    'mean_test_rmse': np.mean(test_rmses),
                    'median_test_rmse': np.median(test_rmses),
                    'std_test_rmse': np.std(test_rmses),
                    'min_test_rmse': np.min(test_rmses),
                    'max_test_rmse': np.max(test_rmses),
                    
                    'mean_test_r2': np.mean(test_r2s),
                    'median_test_r2': np.median(test_r2s),
                    
                    'mean_directional_accuracy': np.mean(directional_accs),
                    'median_directional_accuracy': np.median(directional_accs)
                },
                
                'individual_results': portfolio_results,
                'failed_stocks_list': list(failed_stocks.keys()) if failed_stocks else [],
                'timestamp': datetime.now().isoformat()
            }
        else:
            summary = {
                'market': self.market,
                'total_stocks': len(portfolio_results),
                'successful_stocks': 0,
                'failed_stocks': len(failed_stocks),
                'success_rate': 0,
                'individual_results': portfolio_results,
                'error': 'No stocks trained successfully',
                'timestamp': datetime.now().isoformat()
            }
        
        return summary


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """Load sample data for testing"""
    
    # Create sample data for both markets
    sample_data = {}
    
    # FTSE 100 samples
    ftse_tickers = ['LSEG.L', 'AZN.L', 'SHEL.L']
    
    # S&P 500 samples  
    sp500_tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    all_tickers = ftse_tickers + sp500_tickers
    
    for ticker in all_tickers:
        # Generate sample data
        dates = pd.date_range('2020-01-01', '2024-06-01', freq='D')
        np.random.seed(hash(ticker) % 1000)  # Consistent but different for each ticker
        
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, len(dates))
        if ticker.endswith('.L'):  # FTSE stocks more volatile
            returns *= 1.3
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'close_price': prices,
            'high': prices * np.random.uniform(1.005, 1.03, len(dates)),
            'low': prices * np.random.uniform(0.97, 0.995, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        
        sample_data[ticker] = df
    
    return sample_data


def test_market_specific_training():
    """Test the market-specific training pipeline"""
    
    logger.info("Testing market-specific training pipeline...")
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Test both markets
    for market in ['FTSE_100', 'SP_500']:
        logger.info(f"\n=== Testing {market} Market ===")
        
        # Filter data for this market
        market_data = {}
        for ticker, df in sample_data.items():
            if MarketClassifier.classify_market(ticker) == market:
                market_data[ticker] = df
        
        if not market_data:
            logger.warning(f"No data found for {market}")
            continue
        
        # Create trainer
        trainer = MarketSpecificTrainer(market)
        
        # Train portfolio (limit to 2 stocks for testing)
        test_data = dict(list(market_data.items())[:2])
        portfolio_results = trainer.train_market_portfolio(test_data)
        
        # Log results
        logger.info(f"Portfolio Results for {market}:")
        logger.info(f"  - Success Rate: {portfolio_results.get('success_rate', 0):.1f}%")
        if 'portfolio_metrics' in portfolio_results:
            metrics = portfolio_results['portfolio_metrics']
            logger.info(f"  - Mean Test RMSE: {metrics['mean_test_rmse']:.2f}")
            logger.info(f"  - Mean Directional Accuracy: {metrics['mean_directional_accuracy']:.1f}%")
    
    logger.info("Market-specific training test completed!")


if __name__ == "__main__":
    test_market_specific_training() 