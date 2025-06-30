"""
Standalone script to test connections to all external data sources.
"""
import os
import sys
import logging
import yfinance as yf
from dotenv import load_dotenv

# --- Setup Paths and Logging ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
load_dotenv(os.path.join(project_root, '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the collectors and config
try:
    from data_acquisition import EconomicDataCollector, SentimentDataCollector
    from config import FRED_API_KEY, EIA_API_KEY, TWITTER_API_KEY, REDDIT_CLIENT_ID, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    IMPORTS_SUCCESSFUL = False

def check_yfinance():
    """Tests the yfinance connection."""
    logging.info("--- 1. Testing Financial Data (Yahoo Finance) ---")
    try:
        data = yf.download('AAPL', period='1d', progress=False)
        if not data.empty:
            logging.info("✅ SUCCESS: yfinance connection is working.")
            return True
        else:
            logging.warning("⚠️ WARNING: yfinance connected but returned no data for AAPL.")
            return False
    except Exception as e:
        logging.error(f"❌ FAILURE: yfinance connection failed: {e}", exc_info=True)
        return False

def check_fred():
    """Tests the FRED API connection."""
    logging.info("--- 2. Testing Economic Data (FRED) ---")
    if not FRED_API_KEY:
        logging.warning("⚠️ SKIPPED: FRED_API_KEY is not set in the environment.")
        return None
    try:
        eco_collector = EconomicDataCollector()
        data = eco_collector.get_fred_data('FEDFUNDS')
        if data is not None and not data.empty:
            logging.info("✅ SUCCESS: FRED API connection is working.")
            return True
        else:
            logging.error("❌ FAILURE: FRED connection failed. The API returned no data.")
            return False
    except Exception as e:
        logging.error(f"❌ FAILURE: FRED connection failed with an exception: {e}", exc_info=True)
        return False

def check_eia():
    """Tests the EIA API connection."""
    logging.info("--- 3. Testing Economic Data (EIA) ---")
    if not EIA_API_KEY or EIA_API_KEY == "DEMO_KEY":
        logging.warning("⚠️ SKIPPED: EIA_API_KEY is not set or is a demo key.")
        return None
    try:
        eco_collector = EconomicDataCollector()
        data = eco_collector.get_eia_data('PET.RBRTE.D')
        if data is not None and not data.empty:
             logging.info("✅ SUCCESS: EIA API connection is working.")
             return True
        else:
             logging.error("❌ FAILURE: EIA connection failed. The API returned no data.")
             return False
    except Exception as e:
        logging.error(f"❌ FAILURE: EIA connection failed with an exception: {e}", exc_info=True)
        return False

def check_twitter():
    """Tests the Twitter/X API connection."""
    logging.info("--- 4. Testing Sentiment Data (Twitter/X) ---")
    if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET]):
        logging.warning("⚠️ SKIPPED: One or more Twitter API keys are missing.")
        return None
    try:
        import tweepy
        client = tweepy.Client(
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
        )
        # The new version of the API doesn't have a simple verify_credentials().
        # We can try a basic, low-cost API call like getting our own user info.
        response = client.get_me()
        if response.data:
            logging.info(f"✅ SUCCESS: Twitter client authenticated as @{response.data.username}.")
            return True
        else:
             logging.error("❌ FAILURE: Twitter authentication failed. Response did not contain data. Check keys and app permissions.")
             return False
    except Exception as e:
        logging.error(f"❌ FAILURE: Twitter connection failed with an exception: {e}", exc_info=True)
        return False

def check_reddit():
    """Tests the Reddit API connection."""
    logging.info("--- 5. Testing Sentiment Data (Reddit) ---")
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        logging.warning("⚠️ SKIPPED: One or more Reddit API keys are missing.")
        return None
    try:
        import praw
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        # Check if connection is read-only
        if reddit.read_only:
             logging.info("✅ SUCCESS: Reddit client initialized in read-only mode.")
             return True
        else:
             # If not read-only, we should have a user.
             if reddit.user.me():
                 logging.info(f"✅ SUCCESS: Reddit client authenticated as u/{reddit.user.me().name}.")
                 return True
             else:
                 logging.error("❌ FAILURE: Reddit client failed to authenticate a user. Check credentials if you expect write access.")
                 return False
    except Exception as e:
        logging.error(f"❌ FAILURE: Reddit connection failed with an exception: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if not IMPORTS_SUCCESSFUL:
        print("\nCould not run tests because of an import error. Please check the logs.")
    else:
        results = {
            "Yahoo Finance": check_yfinance(),
            "FRED (Economic)": check_fred(),
            "EIA (Economic)": check_eia(),
            "Twitter/X (Sentiment)": check_twitter(),
            "Reddit (Sentiment)": check_reddit(),
        }
        
        print("\n" + "="*50)
        print("      Data Source Connection Test Summary")
        print("="*50)
        for source, status in results.items():
            if status is True:
                print(f"  ✅ {source:<25} - PASSED")
            elif status is False:
                print(f"  ❌ {source:<25} - FAILED")
            else: # None
                print(f"  ⚠️ {source:<25} - SKIPPED (No API Key)")
        print("="*50)
        print("\nReview the log output above for details on any failures.")
