"""
LSTM model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseForecastModel


class LSTMModel(BaseForecastModel):
    """
    LSTM (Long Short-Term Memory) model for time series forecasting.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 units: int = 50,
                 layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length (int): Number of time steps to look back
            units (int): Number of LSTM units
            layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            learning_rate (float): Learning rate
            **kwargs: Additional parameters
        """
        super().__init__("LSTM", 
                        sequence_length=sequence_length,
                        units=units,
                        layers=layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        **kwargs)
        
        self.sequence_length = sequence_length
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Initialize scaler
        self.scaler = MinMaxScaler()
        
        # Initialize model
        self.model = self._build_model()
    
    def _build_model(self) -> Sequential:
        """
        Build LSTM model architecture.
        
        Returns:
            Sequential: Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.units, 
                      return_sequences=True, 
                      input_shape=(self.sequence_length, 1)))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(self.layers - 1):
            model.add(LSTM(units=self.units, return_sequences=True))
            model.add(Dropout(self.dropout))
        
        # Final LSTM layer
        model.add(LSTM(units=self.units, return_sequences=False))
        model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                     loss='mse')
        
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Features and target
        """
        # Ensure date column is datetime
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        df = df.sort_values(self.date_column)
        
        # Get target series
        target_series = df[self.target_column].dropna()
        
        # Reshape for scaler
        target_reshaped = target_series.values.reshape(-1, 1)
        
        # Scale the data
        target_scaled = self.scaler.fit_transform(target_reshaped)
        
        return df, target_scaled
    
    def fit(self, df: pd.DataFrame, 
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32) -> 'LSTMModel':
        """
        Fit the LSTM model.
        
        Args:
            df (pd.DataFrame): Training data
            validation_split (float): Validation split ratio
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            LSTMModel: Self for chaining
        """
        logger.info("Fitting LSTM model")
        
        # Prepare data
        _, target_scaled = self.prepare_data(df)
        
        if len(target_scaled) < self.sequence_length + 10:
            raise ValueError("Insufficient data for LSTM model")
        
        # Create sequences
        X, y = self._create_sequences(target_scaled)
        
        # Reshape X for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split into training and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store training history
            self.training_history = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'n_observations': len(target_scaled),
                'sequence_length': self.sequence_length
            }
            
            self.is_fitted = True
            logger.info("LSTM model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            raise
        
        return self
    
    def predict(self, df: pd.DataFrame, steps: int = 30) -> np.ndarray:
        """
        Make predictions with LSTM model.
        
        Args:
            df (pd.DataFrame): Input data
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare data
        _, target_scaled = self.prepare_data(df)
        
        # Get the last sequence
        last_sequence = target_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        
        # Generate predictions step by step
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(last_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        logger.info(f"LSTM forecast generated for {steps} steps")
        return predictions.flatten()
    
    def predict_in_sample(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make in-sample predictions.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: In-sample predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        _, target_scaled = self.prepare_data(df)
        X, _ = self._create_sequences(target_scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        # Pad with NaNs to match original length
        return np.concatenate([
            np.full(self.sequence_length, np.nan),
            predictions.flatten()
        ])
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting history")
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.training_history['loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss difference
        loss_diff = np.array(self.training_history['loss']) - np.array(self.training_history['val_loss'])
        ax2.plot(loss_diff, label='Loss Difference (Train - Val)')
        ax2.set_title('Loss Difference')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def get_model_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            str: Model summary
        """
        if not self.is_fitted:
            return "Model not fitted"
        
        # Capture model summary
        from io import StringIO
        
        summary_io = StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        summary = summary_io.getvalue()
        summary_io.close()
        
        return summary 