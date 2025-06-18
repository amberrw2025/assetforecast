import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import streamlit as st

def prepare_features(df, lookback=7):
    """Prepare features for prediction models."""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure we have enough data points (need at least lookback + 1 points)
    if len(df) < lookback + 1:
        raise ValueError(f"Not enough data points. Need at least {lookback + 1} points, got {len(df)}")
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Add more features with forward fill for NaN values
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std()
    
    # Forward fill NaN values instead of dropping them
    df = df.ffill()
    
    # If there are still any NaN values (at the start), fill them with the first valid value
    df = df.bfill()
    
    # Double check for any remaining NaN values and fill with 0
    df = df.fillna(0)
    
    # Check if we still have enough data after processing
    if len(df) < lookback + 1:
        raise ValueError(f"Not enough data points after processing. Need at least {lookback + 1} points, got {len(df)}")
    
    # Prepare features
    X = []
    y = []
    
    for i in range(lookback, len(df)):
        # Create feature vector using past prices and technical indicators
        price_features = df['Close'].iloc[i-lookback:i].values
        ma5 = df['MA5'].iloc[i-1]
        ma20 = df['MA20'].iloc[i-1]
        volatility = df['Volatility'].iloc[i-1]
        
        # Reshape price_features to 1D array and concatenate with other features
        features = np.concatenate([price_features.flatten(), [ma5, ma20, volatility]])
        X.append(features)
        y.append(df['Close'].iloc[i])
    
    # Convert to numpy arrays and ensure proper shapes
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # Reshape y to 2D array
    
    # Validate data shapes and check for NaN values
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid data points after preparation")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values found in prepared data")
    
    return X, y, df

def calculate_combined_prediction(predictions, confidence_intervals):
    """
    Calculate a weighted average prediction based on model confidence intervals.
    
    Args:
        predictions (dict): Dictionary of model predictions
        confidence_intervals (dict): Dictionary of model confidence intervals
        
    Returns:
        tuple: (combined_prediction, combined_interval)
    """
    if not predictions or not confidence_intervals:
        return None, None
        
    # Calculate weights based on inverse of prediction interval width
    weights = {}
    total_weight = 0
    
    for model in predictions:
        if model in confidence_intervals:
            lower, upper = confidence_intervals[model]
            interval_width = upper - lower
            if interval_width > 0:  # Avoid division by zero
                weights[model] = 1 / interval_width
                total_weight += weights[model]
    
    if total_weight == 0:
        return None, None
    
    # Normalize weights
    for model in weights:
        weights[model] /= total_weight
    
    # Calculate weighted prediction
    combined_pred = sum(predictions[model] * weights[model] for model in weights)
    
    # Calculate combined interval
    combined_lower = sum(confidence_intervals[model][0] * weights[model] for model in weights)
    combined_upper = sum(confidence_intervals[model][1] * weights[model] for model in weights)
    
    return combined_pred, (combined_lower, combined_upper)

def train_and_predict(df, model_type='linear', params=None):
    """
    Train models and predict the next day's price.
    
    Args:
        df (pd.DataFrame): DataFrame containing Bitcoin price data
        model_type (str): 'linear', 'random_forest', 'xgboost', or 'all'
        params (dict): Model parameters
        
    Returns:
        tuple: (models, predictions, confidence_intervals)
    """
    if df is None or len(df) < 30:
        return None, None, None
    
    # Default parameters
    default_params = {
        'lookback': 7,
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Update parameters if provided
    if params:
        default_params.update(params)
    
    # Prepare features
    X, y, df = prepare_features(df, default_params['lookback'])
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    predictions = {}
    confidence_intervals = {}
    
    # Train Linear Regression
    if model_type in ['linear', 'all']:
        lr_model = LinearRegression()
        lr_model.fit(X_scaled, y)
        models['linear'] = lr_model
        
        # Make prediction for next day
        last_data = X_scaled[-1].reshape(1, -1)
        lr_prediction = lr_model.predict(last_data)[0].item()
        predictions['linear'] = lr_prediction
        
        # Calculate confidence interval
        y_pred = lr_model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        std = np.sqrt(mse)
        confidence_intervals['linear'] = (lr_prediction - 1.96*std.item(), lr_prediction + 1.96*std.item())
    
    # Train Random Forest
    if model_type in ['random_forest', 'all']:
        rf_model = RandomForestRegressor(
            n_estimators=default_params['n_estimators'],
            max_depth=default_params['max_depth'],
            random_state=42
        )
        rf_model.fit(X_scaled, y)
        models['random_forest'] = rf_model
        
        # Make prediction for next day
        last_data = X_scaled[-1].reshape(1, -1)
        rf_prediction = rf_model.predict(last_data)[0].item()
        predictions['random_forest'] = rf_prediction
        
        # Calculate confidence interval using Random Forest's built-in prediction intervals
        predictions_all = np.array([tree.predict(last_data) for tree in rf_model.estimators_])
        confidence_intervals['random_forest'] = (
            np.percentile(predictions_all, 2.5),
            np.percentile(predictions_all, 97.5)
        )
    
    # Train XGBoost
    if model_type in ['xgboost', 'all']:
        xgb_model = xgb.XGBRegressor(
            n_estimators=default_params['n_estimators'],
            max_depth=default_params['max_depth'],
            learning_rate=default_params['learning_rate'],
            subsample=default_params['subsample'],
            colsample_bytree=default_params['colsample_bytree'],
            random_state=42
        )
        xgb_model.fit(X_scaled, y)
        models['xgboost'] = xgb_model
        
        # Make prediction for next day
        last_data = X_scaled[-1].reshape(1, -1)
        xgb_prediction = xgb_model.predict(last_data)[0].item()
        predictions['xgboost'] = xgb_prediction
        
        # Calculate confidence interval using XGBoost's built-in prediction intervals
        # Get predictions from each tree
        booster = xgb_model.get_booster()
        dtest = xgb.DMatrix(last_data)
        predictions_all = []
        for i in range(xgb_model.n_estimators):
            pred = booster.predict(dtest, iteration_range=(i, i+1))
            predictions_all.append(pred[0])
        predictions_all = np.array(predictions_all)
        confidence_intervals['xgboost'] = (
            np.percentile(predictions_all, 2.5),
            np.percentile(predictions_all, 97.5)
        )
    
    # Calculate combined prediction if multiple models are used
    if model_type == 'all' and len(predictions) > 1:
        combined_pred, combined_interval = calculate_combined_prediction(predictions, confidence_intervals)
        predictions['combined'] = combined_pred
        confidence_intervals['combined'] = combined_interval
    
    return models, predictions, confidence_intervals 

def generate_future_predictions(df, models, predictions, confidence_intervals, days_ahead=180):
    """Generate future predictions for visualization."""
    if not models or not predictions:
        return None, None, None

    # Get the last date from historical data
    last_date = pd.to_datetime(df.index[-1])
    
    # Generate future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    
    # Initialize dictionaries for future predictions and intervals
    future_predictions = {model: [] for model in models}
    future_predictions['combined'] = []
    future_intervals = {model: [] for model in models}
    future_intervals['combined'] = []
    
    # Get the lookback period from the trained model
    # The number of features is lookback + 3 (ma5, ma20, volatility)
    lookback_period = models[list(models.keys())[0]].n_features_in_ - 3
    last_data = df.copy()

    # Initialize scaler
    scaler = StandardScaler()
    
    # For each future day
    for _ in range(days_ahead):
        # Prepare features from the last available data
        X, y, _ = prepare_features(last_data, lookback_period)
        if X.shape[0] == 0:
            break
            
        X_scaled = scaler.fit_transform(X)
        last_X_scaled = X_scaled[-1].reshape(1, -1)

        current_predictions = {}
        current_intervals = {}

        # Make predictions for each model
        for model_name, model in models.items():
            pred = model.predict(last_X_scaled)[0]
            if isinstance(pred, np.ndarray):
                pred = pred.item()
            future_predictions[model_name].append(pred)
            current_predictions[model_name] = pred
            
            # Calculate confidence intervals
            if model_name == 'linear':
                y_pred_all = model.predict(X_scaled)
                mse = np.mean((y - y_pred_all) ** 2)
                std = np.sqrt(mse)
                current_intervals[model_name] = (pred - 1.96 * std, pred + 1.96 * std)
            elif model_name == 'random_forest':
                preds = np.array([tree.predict(last_X_scaled) for tree in model.estimators_])
                current_intervals[model_name] = (np.percentile(preds, 2.5), np.percentile(preds, 97.5))
            elif model_name == 'xgboost':
                booster = model.get_booster()
                dtest = xgb.DMatrix(last_X_scaled)
                preds = []
                for i in range(model.n_estimators):
                    tree_pred = booster.predict(dtest, iteration_range=(i, i + 1))
                    preds.append(tree_pred[0])
                current_intervals[model_name] = (np.percentile(preds, 2.5), np.percentile(preds, 97.5))
            
            future_intervals[model_name].append(current_intervals[model_name])

        # Calculate combined prediction
        combined_pred, combined_interval = calculate_combined_prediction(current_predictions, current_intervals)
        
        if combined_pred is not None:
            future_predictions['combined'].append(combined_pred)
            future_intervals['combined'].append(combined_interval)
            next_price = combined_pred
        else:
            # Fallback to average if combined fails
            next_price = np.mean(list(current_predictions.values()))
            future_predictions['combined'].append(next_price)
            # Create a placeholder interval
            lower = np.mean([i[0] for i in current_intervals.values()])
            upper = np.mean([i[1] for i in current_intervals.values()])
            future_intervals['combined'].append((lower, upper))


        # Append the new prediction to the dataframe to be used in the next iteration
        last_date += pd.Timedelta(days=1)
        new_row = pd.DataFrame({'Close': [next_price]}, index=[last_date])
        last_data = pd.concat([last_data, new_row])

        # Recalculate features for the updated dataframe
        last_data['Returns'] = last_data['Close'].pct_change()
        last_data['MA5'] = last_data['Close'].rolling(window=5, min_periods=1).mean()
        last_data['MA20'] = last_data['Close'].rolling(window=20, min_periods=1).mean()
        last_data['Volatility'] = last_data['Returns'].rolling(window=20, min_periods=1).std()
        last_data.ffill(inplace=True)
        last_data.bfill(inplace=True)

    # Convert lists to numpy arrays
    for model in future_predictions:
        future_predictions[model] = np.array(future_predictions[model])
        future_intervals[model] = np.array(future_intervals[model])

    return future_dates, future_predictions, future_intervals 