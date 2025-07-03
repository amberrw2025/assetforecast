# ========================================================================================
# EVALUATION CONFIGURATION
# ========================================================================================

# Primary evaluation metric (changed from RMSE to MAPE for better stock forecasting)
PRIMARY_EVALUATION_METRIC = 'mape'  # Mean Absolute Percentage Error
SECONDARY_METRICS = ['rmse', 'mae', 'r2', 'directional_accuracy']

# Metric optimization direction (lower is better for MAPE, RMSE, MAE; higher is better for R²)
METRIC_OPTIMIZATION = {
    'mape': 'minimize',      # Primary metric - percentage error
    'rmse': 'minimize',      # Raw error magnitude  
    'mae': 'minimize',       # Mean absolute error
    'r2': 'maximize',        # Coefficient of determination
    'directional_accuracy': 'maximize'  # Trading direction accuracy
}

# Performance thresholds for different metrics
PERFORMANCE_THRESHOLDS = {
    'mape': {
        'excellent': 5.0,    # < 5% error
        'good': 10.0,        # < 10% error  
        'acceptable': 15.0,  # < 15% error
        'poor': 25.0         # > 25% error
    },
    'rmse': {
        'ftse_excellent': 50.0,   # < £50 for FTSE
        'ftse_acceptable': 100.0,  # < £100 for FTSE
        'sp500_excellent': 15.0,   # < $15 for S&P 500
        'sp500_acceptable': 30.0   # < $30 for S&P 500
    },
    'directional_accuracy': {
        'excellent': 65.0,    # > 65% direction accuracy
        'good': 55.0,         # > 55% direction accuracy
        'acceptable': 50.0    # > 50% direction accuracy (random)
    }
}

# Evaluation reporting preferences
EVALUATION_DISPLAY = {
    'primary_metric': PRIMARY_EVALUATION_METRIC,
    'precision': {
        'mape': 2,      # 2 decimal places for MAPE (e.g., 5.23%)
        'rmse': 2,      # 2 decimal places for RMSE
        'mae': 2,       # 2 decimal places for MAE
        'r2': 3,        # 3 decimal places for R² (e.g., 0.842)
        'directional_accuracy': 1  # 1 decimal place (e.g., 65.2%)
    },
    'sort_ascending': True,  # True for MAPE (lower is better), False for R² (higher is better)
    'include_rmse_for_comparison': True  # Keep RMSE for legacy comparison
} 