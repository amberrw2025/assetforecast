import yfinance as yf
import logging

# --- Setup Logging ---
# This will print log messages to the console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ticker(ticker):
    """
    Attempts to fetch data for a single ticker and prints the result or error.
    """
    logging.info(f"--- Attempting to fetch data for {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        
        # Try to get historical data
        hist = stock.history(period="1mo")
        
        if hist.empty:
            logging.warning(f"No historical data returned for {ticker}. The ticker might be invalid or delisted.")
            logging.info(f"Attempting to get company info for {ticker} as a fallback...")
            info = stock.info
            # Check if info dict has a key that indicates it's a valid ticker
            if info and info.get('regularMarketPrice') is not None:
                 logging.info(f"Successfully fetched .info object for {ticker}. Market price is {info.get('regularMarketPrice')}")
            else:
                 logging.error(f"Failed to get any data for {ticker}. The .info object is also empty or incomplete.")
        else:
            logging.info(f"Successfully fetched {len(hist)} rows of historical data for {ticker}.")
            print("Data Head:")
            print(hist.head())

    except Exception as e:
        logging.error(f"An exception occurred while fetching data for {ticker}:")
        logging.error(e, exc_info=True) # Print the full traceback

if __name__ == "__main__":
    print("Running yfinance connection test...")
    
    # Test a common US stock
    test_ticker("AAPL") 
    
    print("-" * 50)
    
    # Test the problematic LSE stock
    test_ticker("AZN.L")
    
    print("\nTest complete.") 