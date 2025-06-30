"""
Financial data acquisition module for the Forecast Accuracy Assessment Model.
Handles collection of company financial data using yfinance and other sources.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm

from config import COMPANIES, FINANCIAL_METRICS, DATA_SOURCES
from utils.logger import get_logger
from utils.decorators import retry

logger = get_logger("financial_data")

class FinancialDataCollector:
    """
    Collects financial data for selected companies using yfinance and other sources.
    """
    
    def __init__(self):
        self.start_date = DATA_SOURCES["start_date"]
        self.end_date = DATA_SOURCES.get("end_date") or datetime.now().strftime("%Y-%m-%d")
        
    @retry((ValueError, Exception), tries=3, delay=5, backoff=2)
    def get_company_info(self, ticker: str) -> Dict:
        """
        Fetch basic company information from yfinance.
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            Dict: Company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check for empty info dictionary
            if not info or info.get('trailingPegRatio') is None: # trailingPegRatio is a canary for empty/bad data
                raise ValueError(f"No data returned for {ticker}")

            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", None),
                "currency": info.get("currency", "USD")
            }
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {str(e)}")
            # Re-raise the exception to trigger the retry decorator
            raise e
    
    @retry((ValueError, Exception), tries=3, delay=5, backoff=2)
    def get_financial_metrics(self, ticker: str) -> pd.DataFrame:
        """
        Fetch financial metrics for a specific company.
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            pd.DataFrame: Financial metrics over time
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(start=self.start_date, end=self.end_date)
            if hist.empty:
                raise ValueError(f"No historical data found for {ticker}")
            logger.info(f"Historical data for {ticker}: {hist.head()}")
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Get company info for static metrics
            info = stock.info
            if not info or info.get('trailingPegRatio') is None:
                raise ValueError(f"No info data returned for {ticker}")
            logger.info(f"Info for {ticker}: {info}")
            
            # Create metrics dictionary
            metrics = {
                "date": hist.index,
                "ticker": ticker,
                "close_price": hist["Close"],
                "volume": hist["Volume"],
                "high": hist["High"],
                "low": hist["Low"],
                "open_price": hist["Open"]
            }
            
            # Add static financial metrics (these don't change daily)
            if financials is not None and not financials.empty:
                # Get most recent values
                latest_financials = financials.iloc[:, 0]  # Most recent quarter
                
                # Add to metrics (will be forward-filled for daily data)
                metrics["revenue"] = latest_financials.get("Total Revenue", None)
                metrics["net_income"] = latest_financials.get("Net Income", None)
                metrics["operating_income"] = latest_financials.get("Operating Income", None)
            
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                metrics["total_assets"] = latest_balance.get("Total Assets", None)
                metrics["total_liabilities"] = latest_balance.get("Total Liabilities", None)
                metrics["total_equity"] = latest_balance.get("Total Stockholder Equity", None)
            
            if cashflow is not None and not cashflow.empty:
                latest_cashflow = cashflow.iloc[:, 0]
                metrics["operating_cash_flow"] = latest_cashflow.get("Operating Cash Flow", None)
                metrics["free_cash_flow"] = latest_cashflow.get("Free Cash Flow", None)
            
            # Add info-based metrics
            metrics["profit_margin"] = info.get("profitMargins", None)
            metrics["headcount"] = info.get("fullTimeEmployees", None)
            metrics["market_cap"] = info.get("marketCap", None)
            metrics["pe_ratio"] = info.get("trailingPE", None)
            metrics["pb_ratio"] = info.get("priceToBook", None)
            
            # Create DataFrame
            df = pd.DataFrame(metrics)
            
            # Forward fill static metrics for daily data
            static_columns = ["revenue", "net_income", "operating_income", "total_assets", 
                            "total_liabilities", "total_equity", "operating_cash_flow", 
                            "free_cash_flow", "profit_margin", "headcount", "market_cap", 
                            "pe_ratio", "pb_ratio"]
            
            for col in static_columns:
                if col in df.columns:
                    df[col] = df[col].ffill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {ticker}: {str(e)}")
            # Re-raise exception to allow retry decorator to catch it
            raise e
    
    def collect_all_companies_data(self) -> pd.DataFrame:
        """
        Collect financial data for all selected companies.
        
        Returns:
            pd.DataFrame: Combined financial data for all companies
        """
        all_data = []
        
        # Collect data for all companies
        for market, companies in COMPANIES.items():
            for category, tickers in companies.items():
                logger.info(f"Collecting data for {market} {category}: {tickers}")
                
                for ticker in tqdm(tickers, desc=f"Processing {market} {category}"):
                    try:
                        # Get company info
                        info = self.get_company_info(ticker)
                        
                        # Get financial data
                        financial_data = self.get_financial_metrics(ticker)
                        
                        if not financial_data.empty:
                            # Add metadata
                            financial_data["market"] = market
                            financial_data["category"] = category
                            financial_data["company_name"] = info["name"]
                            financial_data["sector"] = info["sector"]
                            financial_data["industry"] = info["industry"]
                            
                            all_data.append(financial_data)
                        
                        # Rate limiting to avoid API issues
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error processing {ticker} after retries: {str(e)}")
                        continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully collected data for {len(combined_df['ticker'].unique())} companies")
            return combined_df
        else:
            logger.warning("No financial data collected")
            return pd.DataFrame()
    
    def save_financial_data(self, df: pd.DataFrame, filename: str = "company_financials.csv"):
        """
        Save financial data to CSV file.
        
        Args:
            df (pd.DataFrame): Financial data to save
            filename (str): Output filename
        """
        from config import RAW_DATA_DIR
        
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Financial data saved to {filepath}")
        
        return filepath

def main():
    """
    Main function to run financial data collection.
    """
    collector = FinancialDataCollector()
    
    # Collect all company data
    financial_data = collector.collect_all_companies_data()
    
    if not financial_data.empty:
        # Save data
        filepath = collector.save_financial_data(financial_data)
        logger.info(f"Financial data collection completed. Data saved to {filepath}")
        
        # Print summary
        print(f"\nData Collection Summary:")
        print(f"Total records: {len(financial_data)}")
        print(f"Companies: {financial_data['ticker'].nunique()}")
        print(f"Date range: {financial_data['date'].min()} to {financial_data['date'].max()}")
        print(f"Columns: {list(financial_data.columns)}")
    else:
        logger.error("No data collected. Please check your configuration and API access.")

if __name__ == "__main__":
    main() 