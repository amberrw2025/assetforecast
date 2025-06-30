"""
Enhanced LSTM Model with Economic Data Integration
Combines traditional LSTM with economic indicators and technical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import os
import json
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Enhanced LSTM model will not be functional.")

from .base_model import BaseModel
from .economic_data_provider import EconomicDataProvider

class EnhancedLSTMModel(BaseModel):
    """Enhanced LSTM model with economic indicators integration."""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 use_economic_features: bool = True):
        
        super().__init__()
        self.name = "Enhanced LSTM"
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_economic_features = use_economic_features
        
        self.model = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.economic_provider = EconomicDataProvider()
        
        # Feature configuration
        self.economic_features = [
            'fed_funds_rate', 'unemployment_rate', 'treasury_10y', 
            'vix', 'dxy', 'oil_price', 'economic_sentiment'
        ]
        
        self.technical_features = [
            'rsi', 'macd', 'volatility', 'price_to_sma20', 'price_to_sma50'
        ]
        
        self.n_features = 1  # Base price feature
        if self.use_economic_features:
            self.n_features += len(self.economic_features) + len(self.technical_features)
    
    def _prepare_features(self, price_data: pd.Series, target_length: int = None) -> np.ndarray:
        """
        Prepare feature matrix including price, economic, and technical indicators.
        
        Args:
            price_data: Historical price data
            target_length: Target length for features (for prediction)
            
        Returns:
            Feature matrix
        """
        if target_length is None:
            target_length = len(price_data)
        
        # Initialize feature matrix
        features = np.zeros((target_length, self.n_features))
        
        # Price feature (normalized)
        price_array = price_data.values.reshape(-1, 1)
        if len(price_array) >= target_length:
            price_normalized = self.price_scaler.fit_transform(price_array[-target_length:])
            features[:, 0] = price_normalized.flatten()
        else:
            # Handle case where we have less data than target length
            price_normalized = self.price_scaler.fit_transform(price_array)
            features[-len(price_normalized):, 0] = price_normalized.flatten()
        
        if self.use_economic_features:
            try:
                # Get economic indicators
                economic_indicators = self.economic_provider.get_economic_indicators()
                
                # Get technical indicators
                technical_indicators = self.economic_provider.get_technical_indicators(price_data)
                
                # Combine features
                economic_values = [economic_indicators.get(feat, 0.0) for feat in self.economic_features]
                technical_values = [technical_indicators.get(feat, 0.0) for feat in self.technical_features]
                external_features = economic_values + technical_values
                
                # Normalize external features
                external_array = np.array(external_features).reshape(1, -1)
                if hasattr(self.feature_scaler, 'scale_'):
                    external_normalized = self.feature_scaler.transform(external_array)
                else:
                    external_normalized = self.feature_scaler.fit_transform(external_array)
                
                # Broadcast external features to all time steps
                for i in range(len(self.economic_features) + len(self.technical_features)):
                    features[:, i + 1] = external_normalized[0, i]
                    
            except Exception as e:
                print(f"⚠️ Error preparing external features: {e}")
                # Fill with zeros if external features fail
                features[:, 1:] = 0.0
        
        return features
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training/prediction.
        
        Args:
            features: Feature matrix
            target: Target values (for training)
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            if target is not None:
                y.append(target[i])
        
        X = np.array(X)
        y = np.array(y) if target is not None else None
        
        return X, y
    
    def _build_model(self) -> None:
        """Build the enhanced LSTM model architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Enhanced LSTM model")
        
        model = Sequential([
            # First LSTM layer
            LSTM(self.lstm_units, 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dropout(self.dropout_rate / 2),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print(f"✅ Enhanced LSTM model built with {self.n_features} features")
    
    def fit(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Train the enhanced LSTM model.
        
        Args:
            df: DataFrame with 'date' and 'y' columns
            **kwargs: Additional training parameters
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for Enhanced LSTM model")
            
            print(f"🧠 Training Enhanced LSTM model with economic features...")
            
            # Prepare data
            price_data = df['y']
            features = self._prepare_features(price_data)
            
            # Create sequences
            target = self.price_scaler.transform(price_data.values.reshape(-1, 1)).flatten()
            X, y = self._create_sequences(features, target)
            
            if len(X) < 50:
                print(f"⚠️ Insufficient data for LSTM training ({len(X)} sequences)")
                return
            
            # Build model if not exists
            if self.model is None:
                self._build_model()
            
            # Training parameters
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            validation_split = kwargs.get('validation_split', 0.2)
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_fitted = True
            print(f"✅ Enhanced LSTM model trained successfully")
            
            # Store training history
            self.training_history = {
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            print(f"❌ Error training Enhanced LSTM model: {e}")
            self.is_fitted = False
    
    def predict(self, df: pd.DataFrame, steps: int = 30, **kwargs) -> List[float]:
        """
        Generate predictions using the enhanced LSTM model.
        
        Args:
            df: DataFrame with historical data
            steps: Number of steps to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            List of predicted values
        """
        try:
            if not self.is_fitted or self.model is None:
                print("⚠️ Enhanced LSTM model not fitted")
                return self._generate_fallback_forecast(df, steps)
            
            price_data = df['y']
            current_price = price_data.iloc[-1]
            
            # Prepare initial sequence
            features = self._prepare_features(price_data)
            X, _ = self._create_sequences(features)
            
            if len(X) == 0:
                return self._generate_fallback_forecast(df, steps)
            
            # Get last sequence for prediction
            last_sequence = X[-1].reshape(1, self.sequence_length, self.n_features)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Generate predictions iteratively
            for step in range(steps):
                # Predict next value
                pred_normalized = self.model.predict(current_sequence, verbose=0)[0, 0]
                pred_price = self.price_scaler.inverse_transform([[pred_normalized]])[0, 0]
                predictions.append(float(pred_price))
                
                # Update sequence for next prediction
                # Shift sequence and add new prediction
                new_sequence = current_sequence[0, 1:].copy()
                
                # Create new feature vector for the prediction
                extended_price_data = pd.concat([price_data, pd.Series(predictions)])
                new_features = self._prepare_features(extended_price_data, target_length=1)
                new_features[0, 0] = pred_normalized  # Use normalized prediction
                
                # Add new features to sequence
                new_step = new_features.reshape(1, 1, self.n_features)
                current_sequence = np.concatenate([new_sequence, new_step], axis=1).reshape(1, self.sequence_length, self.n_features)
            
            # Apply economic adjustments
            economic_indicators = self.economic_provider.get_economic_indicators()
            technical_indicators = self.economic_provider.get_technical_indicators(price_data)
            
            adjusted_predictions, self.adjustment_details = self.economic_provider.get_forecast_adjustments(
                predictions, economic_indicators, technical_indicators
            )
            
            print(f"✅ Enhanced LSTM prediction completed with economic adjustments")
            return adjusted_predictions
            
        except Exception as e:
            print(f"❌ Error in Enhanced LSTM prediction: {e}")
            return self._generate_fallback_forecast(df, steps)
    
    def _generate_fallback_forecast(self, df: pd.DataFrame, steps: int) -> List[float]:
        """Generate simple trend-based fallback forecast."""
        price_data = df['y']
        last_price = price_data.iloc[-1]
        
        # Simple trend continuation
        recent_change = price_data.pct_change().tail(10).mean()
        
        forecast = []
        current_price = last_price
        
        for i in range(steps):
            # Add some randomness and mean reversion
            change = recent_change * (0.9 ** i) + np.random.normal(0, 0.01)
            current_price *= (1 + change)
            forecast.append(float(current_price))
        
        return forecast
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        info = super().get_model_info()
        
        info.update({
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'use_economic_features': self.use_economic_features,
            'n_features': self.n_features,
            'economic_features': self.economic_features,
            'technical_features': self.technical_features
        })
        
        if hasattr(self, 'training_history'):
            info['training_history'] = self.training_history
            
        if hasattr(self, 'adjustment_details'):
            info['last_adjustment_details'] = self.adjustment_details
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the enhanced LSTM model."""
        if self.model is not None:
            try:
                # Save TensorFlow model
                model_path = Path(filepath).with_suffix('.h5')
                self.model.save(model_path)
                
                # Save scalers and metadata
                metadata = {
                    'model_info': self.get_model_info(),
                    'price_scaler_scale': self.price_scaler.scale_.tolist() if hasattr(self.price_scaler, 'scale_') else None,
                    'price_scaler_min': self.price_scaler.min_.tolist() if hasattr(self.price_scaler, 'min_') else None,
                    'feature_scaler_scale': self.feature_scaler.scale_.tolist() if hasattr(self.feature_scaler, 'scale_') else None,
                    'feature_scaler_min': self.feature_scaler.min_.tolist() if hasattr(self.feature_scaler, 'min_') else None,
                }
                
                metadata_path = Path(filepath).with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✅ Enhanced LSTM model saved to {model_path}")
                
            except Exception as e:
                print(f"❌ Error saving Enhanced LSTM model: {e}")
    
    def load_model(self, filepath: str) -> None:
        """Load the enhanced LSTM model."""
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required to load Enhanced LSTM model")
            
            # Load TensorFlow model
            model_path = Path(filepath).with_suffix('.h5')
            if model_path.exists():
                self.model = tf.keras.models.load_model(model_path)
                
                # Load metadata
                metadata_path = Path(filepath).with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Restore scalers
                    if metadata.get('price_scaler_scale'):
                        self.price_scaler.scale_ = np.array(metadata['price_scaler_scale'])
                        self.price_scaler.min_ = np.array(metadata['price_scaler_min'])
                    
                    if metadata.get('feature_scaler_scale'):
                        self.feature_scaler.scale_ = np.array(metadata['feature_scaler_scale'])
                        self.feature_scaler.min_ = np.array(metadata['feature_scaler_min'])
                
                self.is_fitted = True
                print(f"✅ Enhanced LSTM model loaded from {model_path}")
                
            else:
                print(f"⚠️ Enhanced LSTM model file not found: {model_path}")
                
        except Exception as e:
            print(f"❌ Error loading Enhanced LSTM model: {e}")
            self.is_fitted = False 