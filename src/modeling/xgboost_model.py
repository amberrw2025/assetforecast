import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """Loads processed data from a CSV file."""
    try:
        return pd.read_csv(file_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {file_path}")
        return pd.DataFrame()

def train_xgboost_model(df: pd.DataFrame, target_column: str, feature_columns: list):
    """Trains an XGBoost model using Time Series Cross-Validation."""
    if df.empty:
        return None

    df_clean = df.dropna(subset=feature_columns + [target_column]).copy()
    df_clean = pd.get_dummies(df_clean, columns=['ticker'], drop_first=True)
    
    feature_columns_aligned = [col for col in df_clean.columns if col in feature_columns or col.startswith('ticker_')]
    
    X = df_clean[feature_columns_aligned]
    y = df_clean[target_column]
    
    print("Training XGBoost model with Time Series Cross-Validation...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                             early_stopping_rounds=5, random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    
    # XGBoost requires a slightly different approach for early stopping with CV
    # For simplicity here, we'll use standard cross_val_score and train final model with early stopping
    
    print("Performing cross-validation...")
    cv_scores_r2 = cross_val_score(model, X, y, cv=tscv, scoring='r2', fit_params={'eval_set': [(X, y)]})
    cv_scores_rmse = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', fit_params={'eval_set': [(X, y)]})

    print("\nCross-Validation Scores:")
    print(f"R-squared (R²): {np.mean(cv_scores_r2):.4f} +/- {np.std(cv_scores_r2):.4f}")
    print(f"RMSE: {-np.mean(cv_scores_rmse):.4f} +/- {-np.std(cv_scores_rmse):.4f}")

    print("\nTraining final model on all data...")
    model.fit(X, y, verbose=False)
    print("Final model training complete.")
    
    return model, X.columns

def save_model_and_importance(model, features, model_path: Path, importance_path: Path):
    """Saves the model and its feature importance plot."""
    if model is None:
        return
        
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('XGBoost Feature Importance (Top 20)')
    
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(importance_path)
    print(f"Feature importance plot saved to {importance_path}")
    plt.close()

if __name__ == '__main__':
    PROCESSED_DATA_PATH = Path('data/processed/merged_data.csv')
    df = load_processed_data(PROCESSED_DATA_PATH)
    
    if not df.empty:
        FEATURES = [
            'Open', 'High', 'Low', 'close_price', 'Volume',
            'profit_margins', 'total_revenue', 'full_time_employees',
            'us_interest_rate', 'uk_interest_rate', 'brent_oil_price',
            'us_unemployment_rate', 'vix_volatility_index'
        ]
        TARGET = 'target_price'
        
        xgb_model, feature_names = train_xgboost_model(df, TARGET, FEATURES)
        
        MODEL_SAVE_PATH = Path('models/xgboost_model.joblib')
        IMPORTANCE_PLOT_PATH = Path('reports/figures/xgboost_feature_importance.png')
        save_model_and_importance(xgb_model, feature_names, MODEL_SAVE_PATH, IMPORTANCE_PLOT_PATH) 