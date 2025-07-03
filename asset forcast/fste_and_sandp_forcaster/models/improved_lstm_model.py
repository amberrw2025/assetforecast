"""
Enhanced LSTM Model for Stock Price Forecasting
Addresses critical issues: proper time series validation, enhanced features, market-specific handling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """Enhanced feature engineering for LSTM model"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical features"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Price-based features
        df['returns'] = df['close_price'].pct_change()
        df['log_returns'] = np.log(df['close_price'] / df['close_price'].shift(1))
        
        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'ma_{window}'] = df['close_price'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['close_price'] / df[f'ma_{window}']
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close_price'] / df['close_price'].shift(period) - 1
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'price_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Market regime indicators
        df['trend_strength'] = (df['ma_5'] > df['ma_20']).astype(int)
        df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(60).mean()).astype(int)
        
        return df
    
    def prepare_features_for_lstm(self, df: pd.DataFrame, target_col: str = 'close_price') -> Tuple[np.ndarray, List[str]]:
        """Prepare features specifically for LSTM training"""
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target and date columns
        exclude_cols = [target_col, 'date']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('Unnamed')]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if df[col].isna().sum() / len(df) < 0.3]
        
        # Fill remaining NaN values
        df_features = df[feature_cols].fillna(method='ffill').fillna(0)
        
        self.feature_columns = feature_cols
        return df_features.values, feature_cols

class ImprovedLSTMModel:
    """Enhanced LSTM model with proper time series handling and feature engineering"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [32, 16],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 market_type: str = 'general'):
        """Initialize enhanced LSTM model"""
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.market_type = market_type
        
        # Initialize components
        self.feature_engineer = EnhancedFeatureEngineer()
        self.price_scaler = RobustScaler()
        self.feature_scaler = RobustScaler()
        
        self.model = None
        self.is_fitted = False
        self.training_history = {}
        
        # Apply market-specific configurations
        self._apply_market_specific_config()
    
    def _apply_market_specific_config(self):
        """Apply market-specific configurations"""
        if self.market_type == 'ftse':
            # FTSE 100 tends to be more stable, use lower dropout
            self.dropout_rate = max(0.2, self.dropout_rate - 0.1)
            self.learning_rate = min(0.002, self.learning_rate * 1.5)
        elif self.market_type == 'sp500':
            # S&P 500 more volatile, use higher regularization
            self.dropout_rate = min(0.4, self.dropout_rate + 0.1)
            self.learning_rate = max(0.0005, self.learning_rate * 0.8)
    
    def _build_model(self, n_features: int) -> Model:
        """Build enhanced LSTM architecture"""
        # Price sequence input
        price_input = Input(shape=(self.sequence_length, 1), name='price_sequence')
        
        # Feature sequence input  
        feature_input = Input(shape=(self.sequence_length, n_features), name='feature_sequence')
        
        # LSTM layers for price
        price_lstm = price_input
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            price_lstm = LSTM(units, return_sequences=return_sequences, 
                            name=f'price_lstm_{i}')(price_lstm)
            price_lstm = BatchNormalization()(price_lstm)
            price_lstm = Dropout(self.dropout_rate)(price_lstm)
        
        # LSTM layers for features
        feature_lstm = feature_input
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            feature_lstm = LSTM(units // 2, return_sequences=return_sequences,
                              name=f'feature_lstm_{i}')(feature_lstm)
            feature_lstm = BatchNormalization()(feature_lstm)
            feature_lstm = Dropout(self.dropout_rate)(feature_lstm)
        
        # Combine price and feature representations
        combined = Concatenate()([price_lstm, feature_lstm])
        
        # Dense layers
        dense = combined
        for i, units in enumerate(self.dense_units):
            dense = Dense(units, activation='relu', name=f'dense_{i}')(dense)
            dense = BatchNormalization()(dense)
            dense = Dropout(self.dropout_rate)(dense)
        
        # Output layer
        output = Dense(1, activation='linear', name='price_output')(dense)
        
        # Create model
        model = Model(inputs=[price_input, feature_input], outputs=output)
        
        # Compile with appropriate optimizer
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'close_price') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training with proper time series structure"""
        
        # Create enhanced features
        df_enhanced = self.feature_engineer.create_technical_features(df)
        
        # Prepare features
        features, feature_cols = self.feature_engineer.prepare_features_for_lstm(df_enhanced, target_col)
        
        # Get target values
        target_values = df_enhanced[target_col].values
        
        # Scale the data
        target_scaled = self.price_scaler.fit_transform(target_values.reshape(-1, 1)).flatten()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences
        X_price, X_features, y = [], [], []
        
        for i in range(self.sequence_length, len(target_scaled)):
            # Price sequence (univariate)
            X_price.append(target_scaled[i-self.sequence_length:i])
            
            # Feature sequence (multivariate)
            X_features.append(features_scaled[i-self.sequence_length:i])
            
            # Target
            y.append(target_scaled[i])
        
        X_price = np.array(X_price).reshape(-1, self.sequence_length, 1)
        X_features = np.array(X_features)
        y = np.array(y)
        
        return X_price, X_features, y
    
    def create_time_series_splits(self, X_price: np.ndarray, X_features: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """Create proper time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(X_price):
            splits.append({
                'X_price_train': X_price[train_idx],
                'X_features_train': X_features[train_idx], 
                'y_train': y[train_idx],
                'X_price_val': X_price[val_idx],
                'X_features_val': X_features[val_idx],
                'y_val': y[val_idx]
            })
        
        return splits
    
    def fit(self, df: pd.DataFrame, target_col: str = 'close_price', 
            validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32,
            use_time_series_cv: bool = True) -> 'ImprovedLSTMModel':
        """Fit the enhanced LSTM model with proper time series validation"""
        logger.info(f"Training enhanced LSTM model for {self.market_type} market")
        
        # Prepare sequences
        X_price, X_features, y = self.prepare_sequences(df, target_col)
        
        if len(X_price) < 100:
            raise ValueError(f"Insufficient data for LSTM training. Need at least 100 sequences, got {len(X_price)}")
        
        # Build model
        n_features = X_features.shape[2]
        self.model = self._build_model(n_features)
        
        logger.info(f"Model architecture built with {n_features} features")
        logger.info(f"Input shapes - Price: {X_price.shape}, Features: {X_features.shape}, Target: {y.shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
        ]
        
        if use_time_series_cv:
            # Use time series cross-validation
            splits = self.create_time_series_splits(X_price, X_features, y)
            best_val_loss = float('inf')
            best_weights = None
            
            for i, split in enumerate(splits):
                logger.info(f"Training on time series fold {i+1}/{len(splits)}")
                
                # Reset model weights
                self.model = self._build_model(n_features)
                
                history = self.model.fit(
                    [split['X_price_train'], split['X_features_train']], split['y_train'],
                    validation_data=([split['X_price_val'], split['X_features_val']], split['y_val']),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Track best model
                min_val_loss = min(history.history['val_loss'])
                if min_val_loss < best_val_loss:
                    best_val_loss = min_val_loss
                    best_weights = self.model.get_weights()
                    self.training_history = history.history
            
            # Set best weights
            if best_weights is not None:
                self.model.set_weights(best_weights)
                logger.info(f"Selected best model with validation loss: {best_val_loss:.4f}")
        
        else:
            # Simple train/validation split (time-based)
            split_idx = int(len(X_price) * (1 - validation_split))
            
            X_price_train = X_price[:split_idx]
            X_features_train = X_features[:split_idx]
            y_train = y[:split_idx]
            
            X_price_val = X_price[split_idx:]
            X_features_val = X_features[split_idx:]
            y_val = y[split_idx:]
            
            history = self.model.fit(
                [X_price_train, X_features_train], y_train,
                validation_data=([X_price_val, X_features_val], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
        
        self.is_fitted = True
        logger.info("Enhanced LSTM model training completed successfully")
        
        return self
    
    def predict(self, df: pd.DataFrame, target_col: str = 'close_price', steps: int = 30) -> np.ndarray:
        """Generate predictions using the enhanced LSTM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare features for the input data
        df_enhanced = self.feature_engineer.create_technical_features(df)
        features, _ = self.feature_engineer.prepare_features_for_lstm(df_enhanced, target_col)
        
        # Get and scale target values
        target_values = df_enhanced[target_col].values
        target_scaled = self.price_scaler.transform(target_values.reshape(-1, 1)).flatten()
        features_scaled = self.feature_scaler.transform(features)
        
        # Get last sequence for prediction
        if len(target_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        predictions = []
        
        # Use last sequence as starting point
        current_price_seq = target_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        current_feature_seq = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        for _ in range(steps):
            # Predict next value
            pred_scaled = self.model.predict([current_price_seq, current_feature_seq], verbose=0)[0, 0]
            
            # Inverse transform prediction
            pred_actual = self.price_scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred_actual)
            
            # Update sequences for next prediction
            # Shift price sequence
            current_price_seq = np.roll(current_price_seq, -1, axis=1)
            current_price_seq[0, -1, 0] = pred_scaled
            
            # For features, use the last known feature values
            current_feature_seq = np.roll(current_feature_seq, -1, axis=1)
            current_feature_seq[0, -1, :] = current_feature_seq[0, -2, :]
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save the complete model including scalers and metadata"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        if self.model:
            self.model.save(path / 'lstm_model.h5')
        
        # Save scalers and metadata
        joblib.dump(self.price_scaler, path / 'price_scaler.pkl')
        joblib.dump(self.feature_scaler, path / 'feature_scaler.pkl')
        joblib.dump(self.feature_engineer, path / 'feature_engineer.pkl')
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'market_type': self.market_type,
            'is_fitted': self.is_fitted,
            'training_history': self.training_history
        }
        joblib.dump(config, path / 'config.pkl')
        
        logger.info(f"Enhanced LSTM model saved to {path}")
    
    def load_model(self, path: str):
        """Load the complete model"""
        path = Path(path)
        
        # Load Keras model
        self.model = tf.keras.models.load_model(path / 'lstm_model.h5')
        
        # Load scalers and metadata
        self.price_scaler = joblib.load(path / 'price_scaler.pkl')
        self.feature_scaler = joblib.load(path / 'feature_scaler.pkl')
        self.feature_engineer = joblib.load(path / 'feature_engineer.pkl')
        
        # Load configuration
        config = joblib.load(path / 'config.pkl')
        self.sequence_length = config['sequence_length']
        self.lstm_units = config['lstm_units']
        self.dense_units = config['dense_units']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        self.market_type = config['market_type']
        self.is_fitted = config['is_fitted']
        self.training_history = config['training_history']
        
        logger.info(f"Enhanced LSTM model loaded from {path}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model:
            summary_list = []
            self.model.summary(print_fn=lambda x: summary_list.append(x))
            return '\n'.join(summary_list)
        return "Model not built yet" 