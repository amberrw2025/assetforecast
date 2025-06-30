"""
Data cleaning and preprocessing module for the Forecast Accuracy Assessment Model.
Handles missing values, time alignment, currency standardization, and data merging.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import CLEANING_CONFIG, DATA_SOURCES
from utils.logger import get_logger

logger = get_logger("data_cleaner")

class DataCleaner:
    """
    Handles data cleaning, preprocessing, and integration for the forecast model.
    """
    
    def __init__(self):
        self.cleaning_config = CLEANING_CONFIG
        self.base_currency = DATA_SOURCES["currency"]
        self.frequency = DATA_SOURCES["frequency"]
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Calculate missing value percentages
        missing_pct = df.isnull().sum() / len(df)
        logger.info(f"Missing value percentages:\n{missing_pct[missing_pct > 0]}")
        
        # Remove columns with too many missing values
        threshold = self.cleaning_config["missing_value_threshold"]
        columns_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if columns_to_drop:
            logger.info(f"Dropping columns with >{threshold*100}% missing values: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # Handle remaining missing values based on data type
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric columns
                    if strategy == "auto":
                        # Use forward fill for time series data, then backward fill
                        df[column] = df[column].ffill(limit=self.cleaning_config["forward_fill_limit"])
                        df[column] = df[column].bfill(limit=self.cleaning_config["backward_fill_limit"])
                        
                        # If still missing, use interpolation
                        if df[column].isnull().sum() > 0:
                            df[column] = df[column].interpolate(method=self.cleaning_config["interpolation_method"])
                        
                        # If still missing, use median
                        if df[column].isnull().sum() > 0:
                            df[column] = df[column].fillna(df[column].median())
                    else:
                        df[column] = df[column].fillna(df[column].median())
                        
                elif df[column].dtype == 'object':
                    # Categorical/text columns
                    df[column] = df[column].fillna("Unknown")
                else:
                    # Other types - use forward fill
                    df[column] = df[column].ffill()
        
        logger.info(f"Missing values handled. Remaining missing values: {df.isnull().sum().sum()}")
        return df
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Detecting outliers in columns: {columns}")
        
        threshold = self.cleaning_config["outlier_threshold"]
        
        for column in columns:
            if column in df.columns:
                # Calculate z-scores
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                
                # Find outliers
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in {column}")
                    
                    # Cap outliers at threshold * std
                    upper_bound = df[column].mean() + threshold * df[column].std()
                    lower_bound = df[column].mean() - threshold * df[column].std()
                    
                    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def standardize_time_series(self, df: pd.DataFrame, date_column: str = 'date', 
                               target_frequency: str = None) -> pd.DataFrame:
        """
        Standardize time series frequency and alignment.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Name of the date column
            target_frequency (str): Target frequency for resampling
            
        Returns:
            pd.DataFrame: DataFrame with standardized time series
        """
        if target_frequency is None:
            target_frequency = self.frequency
        
        logger.info(f"Standardizing time series to frequency: {target_frequency}")
        
        # Ensure date column is datetime and handle timezone issues
        try:
            # Convert to datetime, coercing errors, and then remove timezone info
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True)
            df[date_column] = df[date_column].dt.tz_convert(None)
        except Exception as e:
            logger.error(f"Error converting date column: {e}")
            # If there's still an issue, just return the dataframe
            return df
        
        # Sort by date
        df = df.sort_values(date_column)
        
        # Set date as index for resampling
        df_temp = df.set_index(date_column)
        
        # Group by non-date columns and resample
        non_date_columns = [col for col in df.columns if col != date_column]
        
        # For each unique combination of non-date columns, resample
        resampled_dfs = []
        
        if len(non_date_columns) > 0:
            # Group by categorical columns and resample numeric columns
            categorical_cols = df_temp.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_cols:
                for name, group in df_temp.groupby(categorical_cols):
                    if isinstance(name, tuple):
                        group_name = dict(zip(categorical_cols, name))
                    else:
                        group_name = {categorical_cols[0]: name}
                    
                    # Resample numeric columns
                    if numeric_cols:
                        resampled = group[numeric_cols].resample(target_frequency).mean()
                        
                        # Add back categorical columns
                        for col, val in group_name.items():
                            resampled[col] = val
                        
                        resampled_dfs.append(resampled)
            else:
                # No categorical columns, just resample all
                resampled = df_temp.resample(target_frequency).mean()
                resampled_dfs.append(resampled)
        else:
            # Only date column, simple resample
            resampled = df_temp.resample(target_frequency).mean()
            resampled_dfs.append(resampled)
        
        if resampled_dfs:
            # Combine all resampled dataframes
            result = pd.concat(resampled_dfs, ignore_index=False)
            result = result.reset_index()
            
            # Forward fill categorical columns
            categorical_cols = result.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if col in result.columns:
                    result[col] = result[col].ffill()
            
            logger.info(f"Time series standardized. Shape: {result.shape}")
            return result
        else:
            logger.warning("No data to resample")
            return df
    
    def standardize_currency(self, df: pd.DataFrame, currency_column: str = 'currency',
                           value_columns: List[str] = None) -> pd.DataFrame:
        """
        Standardize currency to base currency.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            currency_column (str): Column containing currency information
            value_columns (List[str]): Columns containing monetary values
            
        Returns:
            pd.DataFrame: DataFrame with standardized currency
        """
        logger.info(f"Standardizing currency to {self.base_currency}")
        
        if currency_column not in df.columns:
            logger.warning(f"Currency column '{currency_column}' not found. Skipping currency standardization.")
            return df
        
        if value_columns is None:
            # Auto-detect monetary columns
            monetary_keywords = ['price', 'revenue', 'income', 'assets', 'liabilities', 
                               'cash_flow', 'market_cap', 'value', 'amount']
            value_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in monetary_keywords)]
        
        # Simple currency conversion rates (in practice, you'd use a proper currency API)
        conversion_rates = {
            'USD': 1.0,
            'GBP': 1.25,  # 1 GBP = 1.25 USD (approximate)
            'EUR': 1.10,  # 1 EUR = 1.10 USD (approximate)
            'JPY': 0.007, # 1 JPY = 0.007 USD (approximate)
        }
        
        for column in value_columns:
            if column in df.columns:
                # Create a copy of the column for conversion
                df[f'{column}_{self.base_currency}'] = df[column].copy()
                
                # Apply conversion rates
                for currency, rate in conversion_rates.items():
                    mask = df[currency_column] == currency
                    df.loc[mask, f'{column}_{self.base_currency}'] = df.loc[mask, column] * rate
                
                # Replace original column with converted values
                df[column] = df[f'{column}_{self.base_currency}']
                df = df.drop(columns=[f'{column}_{self.base_currency}'])
        
        # Update currency column
        df[currency_column] = self.base_currency
        
        logger.info(f"Currency standardization completed for columns: {value_columns}")
        return df
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame], 
                      merge_strategy: str = "outer") -> pd.DataFrame:
        """
        Merge multiple datasets based on time and identifiers.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets to merge
            merge_strategy (str): Merge strategy ('inner', 'outer', 'left', 'right')
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        logger.info(f"Merging {len(datasets)} datasets with strategy: {merge_strategy}")
        
        if not datasets:
            logger.error("No datasets provided for merging")
            return pd.DataFrame()
        
        # Start with the first dataset
        merged_df = list(datasets.values())[0].copy()
        
        # Merge with remaining datasets
        for name, dataset in list(datasets.items())[1:]:
            logger.info(f"Merging dataset: {name}")
            
            # Ensure both datasets have a date column
            if 'date' not in merged_df.columns or 'date' not in dataset.columns:
                logger.warning(f"Date column not found in one or both datasets. Skipping merge with {name}")
                continue
            
            # Convert date columns to datetime and ensure consistent timezone handling
            try:
                merged_df['date'] = pd.to_datetime(merged_df['date'])
                dataset['date'] = pd.to_datetime(dataset['date'])
                
                # Convert both to timezone-naive to avoid merge conflicts
                if merged_df['date'].dt.tz is not None:
                    merged_df['date'] = merged_df['date'].dt.tz_convert('UTC').dt.tz_localize(None)
                if dataset['date'].dt.tz is not None:
                    dataset['date'] = dataset['date'].dt.tz_convert('UTC').dt.tz_localize(None)
                    
            except ValueError as e:
                if "Tz-aware datetime.datetime cannot be converted to datetime64" in str(e):
                    # Handle timezone-aware datetime objects by converting to UTC first
                    logger.info("Converting timezone-aware datetime objects to UTC in merge")
                    merged_df['date'] = pd.to_datetime(merged_df['date'], utc=True)
                    dataset['date'] = pd.to_datetime(dataset['date'], utc=True)
                    # Then convert to timezone-naive
                    merged_df['date'] = merged_df['date'].dt.tz_localize(None)
                    dataset['date'] = dataset['date'].dt.tz_localize(None)
                else:
                    raise e
            
            # Merge on date
            merged_df = pd.merge(merged_df, dataset, on='date', how=merge_strategy, suffixes=('', f'_{name}'))
            
            logger.info(f"After merging {name}: shape = {merged_df.shape}")
        
        # Sort by date
        merged_df = merged_df.sort_values('date')
        
        logger.info(f"Final merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for the model.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        logger.info("Creating additional features")
        
        # Ensure date column exists
        if 'date' not in df.columns:
            logger.warning("Date column not found. Skipping feature creation.")
            return df
        
        try:
            df['date'] = pd.to_datetime(df['date'])
        except ValueError as e:
            if "Tz-aware datetime.datetime cannot be converted to datetime64" in str(e):
                # Handle timezone-aware datetime objects by converting to UTC first
                logger.info("Converting timezone-aware datetime objects to UTC in feature creation")
                df['date'] = pd.to_datetime(df['date'], utc=True)
            else:
                raise e
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Only create features for a few key columns to avoid too many NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['year', 'month', 'day_of_week', 'quarter', 'day_of_year']
        
        # Focus on the most important columns
        key_columns = ['close_price', 'volume', 'value']
        selected_columns = [col for col in key_columns if col in numeric_columns]
        
        if not selected_columns:
            # Fallback to first few numeric columns
            selected_columns = [col for col in numeric_columns if col not in exclude_cols][:3]
        
        logger.info(f"Creating features for columns: {selected_columns}")
        
        # Create features more conservatively
        for column in selected_columns:
            if column in df.columns:
                # Only create rolling features (no lag features to avoid NaN issues)
                df[f'{column}_rolling_mean_7'] = df[column].rolling(window=7, min_periods=1).mean()
                df[f'{column}_rolling_std_7'] = df[column].rolling(window=7, min_periods=1).std()
        
        # Don't drop any rows - let the model handle NaN values
        logger.info(f"Feature creation completed. Final shape: {df.shape}")
        return df
    
    def clean_single_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run a standard cleaning pipeline on a single dataset.

        Args:
            df (pd.DataFrame): The input dataframe to be cleaned.

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"Input is not a pandas DataFrame, skipping. Type: {type(df)}")
            return pd.DataFrame()

        # --- 1. De-duplicate data ---
        # Identify date and symbol columns for de-duplication
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        symbol_col = next((c for c in df.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), None)
        
        if date_col and symbol_col:
            initial_rows = len(df)
            df = df.sort_values(by=date_col, ascending=True)
            df.drop_duplicates(subset=[date_col, symbol_col], keep='last', inplace=True)
            removed_count = initial_rows - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate rows based on date and symbol.")
        
        # --- 2. Standardize column names ---
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Rename common variations
        rename_map = {'timestamp': 'date', 'ticker': 'symbol'}
        df = df.rename(columns=rename_map)
        
        # --- 3. Handle missing values ---
        df = self.handle_missing_values(df, strategy=self.cleaning_config.get("missing_value_strategy", "auto"))
        
        # --- 4. Standardize time series ---
        if 'date' in df.columns:
            df = self.standardize_time_series(df, date_column='date')
        
        # --- 5. Handle outliers ---
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df = self.detect_and_handle_outliers(df, columns=numeric_cols)
        
        return df
        
    def merge_datasets(self, datasets: list) -> pd.DataFrame:
        """
        Merge multiple cleaned datasets.

        Args:
            datasets (list): List of cleaned datasets to merge.

        Returns:
            pd.DataFrame: Merged dataset.
        """
        if not datasets:
            return pd.DataFrame()
        
        # Separate financial data from others
        financial_df = None
        other_dfs = []
        for df in datasets:
            if 'close_price' in df.columns:
                financial_df = df
            else:
                other_dfs.append(df)
        
        if financial_df is None:
            # If no financial data, merge what we have
            merged_df = other_dfs[0]
            for df in other_dfs[1:]:
                merged_df = pd.merge(merged_df, df, on='date', how='outer')
            return merged_df

        # Merge other datasets into financial_df
        merged_df = financial_df
        for df in other_dfs:
            merged_df = pd.merge(merged_df, df, on='date', how='outer')
        
        return merged_df

    def clean_and_prepare_data(self, datasets: dict) -> pd.DataFrame:
        """
        Complete data cleaning and preparation pipeline.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets to process
            
        Returns:
            pd.DataFrame: Cleaned and prepared dataset
        """
        logger.info("Starting complete data cleaning and preparation pipeline")

        cleaned_datasets = {}
        for name, dataset in datasets.items():
            logger.info(f"Cleaning dataset: {name}")
            
            # Handle missing values
            dataset = self.handle_missing_values(dataset)
            
            # Handle outliers
            logger.info(f"Handling outliers in dataset: {name}")
            dataset = self.detect_and_handle_outliers(dataset)
            
            # Standardize time series
            if 'date' in dataset.columns:
                 logger.info(f"Standardizing time series for dataset: {name}")
                 dataset = self.standardize_time_series(dataset)

            cleaned_datasets[name] = dataset

        # Merge datasets
        logger.info("Merging cleaned datasets")
        
        # Get financial data if available
        financial_df = cleaned_datasets.pop('financial', None)
        
        if financial_df is None:
            logger.error("No financial data found after cleaning. Cannot proceed.")
            return pd.DataFrame()

        # Merge all other datasets into the financial one
        merged_df = financial_df
        for name, dataset in cleaned_datasets.items():
            if 'date' in dataset.columns:
                merged_df = pd.merge(merged_df, dataset, on='date', how='outer', suffixes=(f'_{name}', ''))
            else:
                logger.warning(f"Dataset '{name}' has no 'date' column, cannot merge.")

        # Final cleaning steps on merged data
        logger.info("Performing final cleaning on merged dataset")
        merged_df = self.handle_missing_values(merged_df)
        merged_df.sort_values(by='date', inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        logger.info(f"Data cleaning and preparation finished. Final shape: {merged_df.shape}")
        return merged_df
    
    def save_cleaned_data(self, df: pd.DataFrame, filename: str = "cleaned_dataset.csv"):
        """
        Save cleaned data to CSV file.
        
        Args:
            df (pd.DataFrame): Cleaned dataset to save
            filename (str): Output filename
        """
        from config import PROCESSED_DATA_DIR
        
        filepath = PROCESSED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Cleaned data saved to {filepath}")
        
        return filepath

def main():
    """
    Main function to demonstrate data cleaning pipeline.
    """
    # This would typically be called with actual datasets
    logger.info("Data cleaning module loaded successfully")
    logger.info("Use DataCleaner.clean_and_prepare_data() to process your datasets")

if __name__ == "__main__":
    main() 