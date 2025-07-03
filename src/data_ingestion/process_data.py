import pandas as pd
from pathlib import Path

def clean_company_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses the raw company data."""
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by ticker and date to ensure correct order for filling
    df.sort_values(by=['ticker', 'date'], inplace=True)
    
    # Forward-fill informational columns that don't change daily
    # (e.g., profit_margins, total_revenue) within each ticker group
    info_cols = ['profit_margins', 'total_revenue', 'full_time_employees']
    df[info_cols] = df.groupby('ticker')[info_cols].ffill()
    
    # For any remaining NaNs after forward-fill (e.g., at the beginning), back-fill
    df[info_cols] = df.groupby('ticker')[info_cols].bfill()

    # Drop rows where essential data like close_price is missing
    df.dropna(subset=['close_price'], inplace=True)
    
    return df

def merge_data(company_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merges company and macroeconomic data based on the date."""
    company_df['date'] = pd.to_datetime(company_df['date'])
    macro_df['date'] = pd.to_datetime(macro_df['date'])

    # Use merge_asof to join the latest macro data to each stock price entry
    # This prevents lookahead bias by matching on or before the date.
    merged_df = pd.merge_asof(
        company_df.sort_values('date'),
        macro_df.sort_values('date'),
        on='date',
        direction='backward' # Finds the most recent macro value on or before the company data date
    )
    return merged_df

def create_target_variable(df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Creates a lagged target variable for forecasting."""
    df.sort_values(by=['ticker', 'date'], inplace=True)
    
    # The target is the closing price 'lag_days' in the future
    df['target_price'] = df.groupby('ticker')['close_price'].shift(-lag_days)
    
    # Remove the last 'lag_days' for each ticker as there is no target
    df.dropna(subset=['target_price'], inplace=True)
    
    return df

if __name__ == '__main__':
    RAW_DATA_DIR = Path('../data/raw')
    PROCESSED_DATA_DIR = Path('../data/processed')
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load raw datasets
        sp500_df = pd.read_csv(RAW_DATA_DIR / 'sp500_raw_data.csv')
        ftse100_df = pd.read_csv(RAW_DATA_DIR / 'ftse100_raw_data.csv')
        macro_df = pd.read_csv(RAW_DATA_DIR / 'macro_raw_data.csv')
        print("Raw datasets loaded successfully.")

        # 2. Combine and clean company data
        company_df = pd.concat([sp500_df, ftse100_df], ignore_index=True)
        company_df_clean = clean_company_data(company_df)
        print("Company data cleaned.")

        # 3. Merge with macro data
        merged_data = merge_data(company_df_clean, macro_df)
        print("Company and macroeconomic data merged.")

        # 4. Create target variable (e.g., predict next day's price)
        final_df = create_target_variable(merged_data, lag_days=1)
        print("Target variable 'target_price' created.")
        
        # 5. Save the processed data
        save_path = PROCESSED_DATA_DIR / 'merged_data.csv'
        final_df.to_csv(save_path, index=False)
        print(f"Processed data saved to {save_path}")
        print(f"Shape of the final dataset: {final_df.shape}")
        print("\nFinal DataFrame columns:")
        print(final_df.info())

    except FileNotFoundError as e:
        print(f"Error: {e}.")
        print("Please ensure the raw data exists by running the ingestion scripts first:")
        print("`python src/data_ingestion/get_company_data.py`")
        print("`python src/data_ingestion/get_macro_data.py`")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 