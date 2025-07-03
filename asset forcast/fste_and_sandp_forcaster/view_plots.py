import streamlit as st
from pathlib import Path
import os
import yfinance as yf

st.set_page_config(layout="wide")

st.title("Forecast Simulation Comparison")
st.write("All generated forecast plots are displayed below for easy comparison.")


# --- Function to get plot files and company names ---
@st.cache_data
def get_plot_data():
    plots_dir = Path(__file__).parent / 'forecast_plots_2024'
    if not plots_dir.exists():
        st.error("Plots directory not found. Please run `generate_forecast_plots.py` first.")
        return {}, {}
    
    plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
    ticker_map = {f.replace('_2024_forecast_simulation.png', ''): f for f in plot_files}

    # Fetch company names
    company_names = {}
    for ticker in ticker_map.keys():
        try:
            stock_info = yf.Ticker(ticker).info
            company_names[ticker] = stock_info.get('longName', ticker)
        except Exception:
            company_names[ticker] = ticker # Fallback to ticker if API fails

    return ticker_map, company_names

# --- Main App ---
ticker_map, company_names = get_plot_data()

if not ticker_map:
    st.stop()

# Define number of columns
num_columns = 2
cols = st.columns(num_columns)

sorted_tickers = sorted(ticker_map.keys())

# Display plots in a grid
for i, ticker in enumerate(sorted_tickers):
    with cols[i % num_columns]:
        plot_path = Path(__file__).parent / 'forecast_plots_2024' / ticker_map[ticker]
        
        # Use the full company name as the subheader
        st.subheader(company_names.get(ticker, ticker))
        st.image(str(plot_path), caption=f"Forecast for {ticker}", use_container_width=True) 