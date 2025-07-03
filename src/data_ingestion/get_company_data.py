import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_initial_company_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches key financial data for a list of stock tickers using yfinance.

    Args:
        tickers: A list of company ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date: The start date for the data in 'YYYY-MM-DD' format.
        end_date: The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        A pandas DataFrame containing the historical stock data.
    """
    all_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Fetch historical market data
            hist_data = stock.history(start=start_date, end=end_date)
            
            # Add ticker symbol for identification
            hist_data['ticker'] = ticker
            
            # Example of getting specific info (some are only available as single points)
            info = stock.info
            hist_data['profit_margins'] = info.get('profitMargins', None)
            hist_data['total_revenue'] = info.get('totalRevenue', None)
            hist_data['full_time_employees'] = info.get('fullTimeEmployees', None)

            all_data.append(hist_data)
            print(f"Successfully fetched data for {ticker}")
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")
            
    if not all_data:
        return pd.DataFrame()

    # Combine data for all tickers
    combined_df = pd.concat(all_data)
    
    # Clean up the dataframe index
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'Date': 'date', 'Close': 'close_price'}, inplace=True)
    
    return combined_df

if __name__ == '__main__':
    # Define tickers for FTSE 100 and S&P 500
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'] # Top 5 by market cap example
    ftse100_tickers = ['AZN.L', 'ULVR.L', 'HSBA.L', 'GLEN.L', 'RIO.L'] # Example tickers

    # The plan starts from 2023
    start = '2023-01-01'
    end = '2023-12-31'
    
    print("Fetching S&P 500 data...")
    sp500_data = fetch_initial_company_data(sp500_tickers, start, end)
    
    print("\nFetching FTSE 100 data...")
    ftse_data = fetch_initial_company_data(ftse100_tickers, start, end)
    
    # Define save paths
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
    
    if not sp500_data.empty:
        sp500_data.to_csv(output_dir / 'sp500_raw_data.csv', index=False)
        print(f"\nS&P 500 data saved to {output_dir / 'sp500_raw_data.csv'}")

    if not ftse_data.empty:
        ftse_data.to_csv(output_dir / 'ftse100_raw_data.csv', index=False)
        print(f"FTSE 100 data saved to {output_dir / 'ftse100_raw_data.csv'}") 