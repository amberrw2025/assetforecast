import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, date
import yfinance as yf

def get_yahoo_data(start_date, end_date):
    """Fetch Bitcoin price data from Yahoo Finance."""
    try:
        print(f"Debug: Starting Yahoo Finance data fetch")
        print(f"Debug: Start date: {start_date}, End date: {end_date}")
        
        # Convert dates to strings if they are date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')
            
        print(f"Debug: Converted dates - Start: {start_date}, End: {end_date}")
            
        # Download BTC-USD data
        print("Debug: Attempting to download BTC-USD data from Yahoo Finance")
        df = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        
        print(f"Debug: Download complete. DataFrame shape: {df.shape if df is not None else 'None'}")
        
        if df is None or df.empty:
            print("Debug: No data received from Yahoo Finance")
            st.warning("No data fetched from Yahoo Finance.")
            return None
            
        # Reset index and rename columns
        print("Debug: Processing DataFrame")
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        # Only keep necessary columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Debug: Final DataFrame shape: {df.shape}")
        print(f"Debug: Date range in data: {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"Debug: Error in get_yahoo_data: {str(e)}")
        st.warning(f"Failed to fetch data from Yahoo Finance: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_btc_data(start_date, end_date):
    """Fetch Bitcoin price data from Yahoo Finance."""
    try:
        print(f"\nDebug: Starting get_btc_data")
        print(f"Debug: Input dates - Start: {start_date}, End: {end_date}")
        
        # Ensure end date is not in the future
        current_date = datetime.now().date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()
        
        if end_date > current_date:
            end_date = current_date
            print(f"Debug: Adjusted end date to current date: {end_date}")
        
        # Convert start_date to date if it's a datetime
        if isinstance(start_date, datetime):
            start_date = start_date.date()
            
        print(f"Debug: Final dates - Start: {start_date}, End: {end_date}")
        
        # Try to get data from Yahoo Finance
        print("Debug: Calling get_yahoo_data")
        df = get_yahoo_data(start_date, end_date)
        
        if df is not None and not df.empty:
            print("Debug: Successfully retrieved data from Yahoo Finance")
            st.success("Successfully fetched data from Yahoo Finance")
            return df
        
        print("Debug: Failed to get data from Yahoo Finance")
        st.error("Failed to fetch data from Yahoo Finance")
        return None
                
    except Exception as e:
        print(f"Debug: Unexpected error in get_btc_data: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")
        return None 