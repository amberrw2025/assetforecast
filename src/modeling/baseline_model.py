import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from pathlib import Path
import joblib
import numpy as np

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Loads processed data from a CSV file."""
    try:
        return pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {file_path}")
        print("Please run a data processing script first.")
        return pd.DataFrame()

def train_baseline_model(df: pd.DataFrame, target_column: str, feature_columns: list):
    """
    Trains a baseline Linear Regression model.
    
    Args:
        df: The input DataFrame containing features and target.
        target_column: The name of the column to predict.
        feature_columns: A list of column names to use as features.
        
    Returns:
        A trained model object and the test data (X_test, y_test) for evaluation.
    """
    if df.empty:
        return None

    # For simplicity, we'll drop rows with any missing values
    df_clean = df.dropna(subset=feature_columns + [target_column]).copy()
    
    # Convert categorical 'ticker' column to numeric using one-hot encoding
    df_clean = pd.get_dummies(df_clean, columns=['ticker'], drop_first=True)
    
    # Align feature columns with the new one-hot encoded columns
    feature_columns_aligned = [col for col in df_clean.columns if col in feature_columns or col.startswith('ticker_')]
    
    X = df_clean[feature_columns_aligned]
    y = df_clean[target_column]
    
    print("Training Linear Regression model with Time Series Cross-Validation...")
    model = LinearRegression()

    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform cross-validation
    cv_scores_r2 = cross_val_score(model, X, y, cv=tscv, scoring='r2')
    cv_scores_rmse = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error')

    print("\nCross-Validation Scores:")
    print(f"R-squared (R²): {np.mean(cv_scores_r2):.4f} +/- {np.std(cv_scores_r2):.4f}")
    print(f"RMSE: {-np.mean(cv_scores_rmse):.4f} +/- {-np.std(cv_scores_rmse):.4f}")

    print("\nTraining final model on all data...")
    model.fit(X, y)
    print("Final model training complete.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """(This function is no longer used with cross-validation but kept for potential simple tests)"""
    if model is None:
        return

    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

def save_model(model, file_path: Path):
    """Saves the trained model to a file."""
    if model is None:
        return
        
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    print(f"\nModel saved to {file_path}")

if __name__ == '__main__':
    # This is a placeholder for the processed data path.
    # Step 4 (Data Cleaning & Integration) would produce this file.
    PROCESSED_DATA_PATH = Path('data/processed/merged_data.csv')

    df = load_processed_data(PROCESSED_DATA_PATH)
    
    if not df.empty:
        # Define features and target based on the merged_data.csv columns
        # We will use the macro data and the previous day's price data as features.
        # Note: 'ticker' will be handled by one-hot encoding.
        FEATURES = [
            'Open', 'High', 'Low', 'close_price', 'Volume', # Stock data
            'profit_margins', 'total_revenue', 'full_time_employees', # Company info
            'us_interest_rate', 'uk_interest_rate', 'brent_oil_price', # Macro data
            'us_unemployment_rate', 'vix_volatility_index'
        ]
        TARGET = 'target_price'
        
        # Train the model using cross-validation
        lr_model = train_baseline_model(df, TARGET, FEATURES)
        
        # Note: Direct evaluation is replaced by cross-validation scores printed during training
        
        # Save the final model
        MODEL_SAVE_PATH = Path('models/linear_regression_baseline.joblib')
        save_model(lr_model, MODEL_SAVE_PATH) 