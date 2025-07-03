import os
import pandas as pd
from fredapi import Fred
from pathlib import Path

def get_fred_data(api_key: str, series_ids: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches macroeconomic data from the FRED API.

    Args:
        api_key: Your FRED API key.
        series_ids: A dictionary mapping series IDs to desired column names.
        start_date: The start date for the data in 'YYYY-MM-DD' format.
        end_date: The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        A pandas DataFrame containing the requested FRED data.
    """
    try:
        fred = Fred(api_key=api_key)
        
        df = pd.DataFrame()
        for series_id, name in series_ids.items():
            series_data = fred.get_series(series_id, start_date, end_date)
            df[name] = series_data
        
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        
        # FRED data can have missing values for non-trading days.
        # We can forward-fill to handle this.
        df.ffill(inplace=True)
        
        return df
    except Exception as e:
        print(f"Failed to fetch FRED data: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # It is recommended to set your FRED_API_KEY as an environment variable
    fred_api_key = os.getenv('FRED_API_KEY')
    
    if not fred_api_key:
        print("FRED_API_KEY environment variable not found.")
        print("Please get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html")
    else:
        # As per the project plan
        series_to_fetch = {
            'FEDFUNDS': 'us_interest_rate',
            'IR3TIB01GBM156N': 'uk_interest_rate',
            'DCOILBRENTEU': 'brent_oil_price', # Using a different series for Brent
            'UNRATE': 'us_unemployment_rate',
            'VIXCLS': 'vix_volatility_index'
        }
        
        start = '2023-01-01'
        end = '2023-12-31'
        
        print("Fetching macroeconomic data from FRED...")
        macro_data = get_fred_data(fred_api_key, series_to_fetch, start, end)
        
        if not macro_data.empty:
            output_dir = Path('data/raw')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = output_dir / 'macro_raw_data.csv'
            macro_data.to_csv(save_path, index=False)
            print(f"Macroeconomic data saved to {save_path}") 