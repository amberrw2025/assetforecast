"""
Sentiment data acquisition module for the Forecast Accuracy Assessment Model.
Handles collection of social media sentiment and news data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from tqdm import tqdm

from config import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, DATA_DIR
from utils.logger import get_logger

logger = get_logger("sentiment_data")

class SentimentDataCollector:
    """
    Collects sentiment data from Twitter and Reddit.
    """
    def __init__(self):
        # Twitter credentials
        self.twitter_api_key = TWITTER_API_KEY
        self.twitter_api_secret = TWITTER_API_SECRET
        self.twitter_access_token = TWITTER_ACCESS_TOKEN
        self.twitter_access_token_secret = TWITTER_ACCESS_TOKEN_SECRET
        
        # Reddit credentials
        self.reddit_client_id = REDDIT_CLIENT_ID
        self.reddit_client_secret = REDDIT_CLIENT_SECRET
        self.reddit_user_agent = REDDIT_USER_AGENT
        
        self.twitter_client = None
        self.reddit_client = None

    def _initialize_twitter(self):
        """Initialize Twitter client if credentials are provided."""
        if not all([self.twitter_api_key, self.twitter_api_secret, 
                    self.twitter_access_token, self.twitter_access_token_secret]):
            logger.warning("Twitter API credentials not found. Skipping Twitter data.")
            self.twitter_client = None
            return

        try:
            import tweepy
            self.twitter_client = tweepy.Client(
                bearer_token=None,  # Adjust if you use a bearer token
                consumer_key=self.twitter_api_key,
                consumer_secret=self.twitter_api_secret,
                access_token=self.twitter_access_token,
                access_token_secret=self.twitter_access_token_secret
            )
            logger.info("Twitter client initialized successfully.")
        except ImportError:
            logger.error("tweepy is not installed. Please install it: pip install tweepy")
            self.twitter_client = None
        except Exception as e:
            logger.error(f"Error initializing Twitter client: {e}")
            self.twitter_client = None

    def _initialize_reddit(self):
        """Initialize Reddit client if credentials are provided."""
        if not all([self.reddit_client_id, self.reddit_client_secret, self.reddit_user_agent]):
            logger.warning("Reddit API credentials not found. Skipping Reddit data.")
            self.reddit_client = None
            return
            
        try:
            import praw
            self.reddit_client = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            logger.info("Reddit client initialized successfully.")
        except ImportError:
            logger.error("praw is not installed. Please install it: pip install praw")
            self.reddit_client = None
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self.reddit_client = None
            
    def get_tweets(self, keyword: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch tweets for a given keyword.
        
        Args:
            keyword (str): Keyword or hashtag to search for
            limit (int): Number of tweets to fetch
            
        Returns:
            pd.DataFrame: DataFrame of tweets
        """
        if not self.twitter_client:
            logger.info(f"Skipping Twitter data for '{keyword}' due to missing client.")
            return pd.DataFrame()

        try:
            tweets = self.twitter_client.search_recent_tweets(
                query=keyword,
                max_results=limit,
                tweet_fields=["created_at", "text", "public_metrics"]
            )
            
            if not tweets.data:
                return pd.DataFrame()
            
            tweet_data = [{
                'date': tweet.created_at,
                'text': tweet.text,
                'source': 'Twitter',
                'keyword': keyword,
                'likes': tweet.public_metrics.get('like_count', 0),
                'retweets': tweet.public_metrics.get('retweet_count', 0)
            } for tweet in tweets.data]
            
            return pd.DataFrame(tweet_data)
            
        except Exception as e:
            logger.error(f"Error fetching tweets for '{keyword}': {e}")
            return pd.DataFrame()

    def get_reddit_posts(self, subreddit: str, keyword: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch Reddit posts from a subreddit.
        
        Args:
            subreddit (str): Subreddit to search in
            keyword (str): Keyword to search for
            limit (int): Number of posts to fetch
            
        Returns:
            pd.DataFrame: DataFrame of Reddit posts
        """
        if not self.reddit_client:
            logger.info(f"Skipping Reddit data for '{keyword}' in r/{subreddit} due to missing client.")
            return pd.DataFrame()
            
        try:
            sub = self.reddit_client.subreddit(subreddit)
            posts = sub.search(keyword, limit=limit)
            
            post_data = [{
                'date': datetime.fromtimestamp(post.created_utc),
                'text': f"{post.title} {post.selftext}",
                'source': 'Reddit',
                'keyword': keyword,
                'score': post.score,
                'comments': post.num_comments
            } for post in posts]
            
            return pd.DataFrame(post_data)
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for '{keyword}': {e}")
            return pd.DataFrame()

    def _generate_sample_social_media_data(self, keyword: str, source: str, limit: int) -> pd.DataFrame:
        """
        Generate sample social media data for testing.
        """
        dates = [datetime.now() - timedelta(days=np.random.randint(0, 30)) for _ in range(limit)]
        sentiments = ["positive", "negative", "neutral"]
        
        data = {
            'date': dates,
            'text': [f"Sample {np.random.choice(sentiments)} text about {keyword}" for _ in range(limit)],
            'source': source,
            'keyword': keyword
        }
        
        return pd.DataFrame(data)

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment scores for text data.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            
        Returns:
            pd.DataFrame: DataFrame with sentiment scores added
        """
        try:
            from textblob import TextBlob
            
            def get_sentiment(text):
                try:
                    blob = TextBlob(str(text))
                    return blob.sentiment.polarity
                except:
                    return 0.0
            
            # Calculate sentiment scores
            df['sentiment_score'] = df['text'].apply(get_sentiment)
            
            # Categorize sentiment
            df['sentiment_category'] = df['sentiment_score'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
            
            logger.info("Sentiment scores calculated successfully")
            return df
            
        except ImportError:
            logger.warning("TextBlob not installed. Skipping sentiment analysis.")
            df['sentiment_score'] = 0.0
            df['sentiment_category'] = 'neutral'
            return df
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {str(e)}")
            df['sentiment_score'] = 0.0
            df['sentiment_category'] = 'neutral'
            return df

    def collect_all_sentiment_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect sentiment data from all sources or use existing data.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of sentiment dataframes
        """
        sentiment_data = {}
        
        # Check if we have existing sentiment data
        sentiment_dir = DATA_DIR / "sentiment"
        if sentiment_dir.exists():
            for file_path in sentiment_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        sentiment_data[file_path.stem] = df
                        logger.info(f"Loaded existing sentiment data from {file_path.name}")
                except Exception as e:
                    logger.warning(f"Could not load {file_path.name}: {str(e)}")
        
        # If no existing data and API keys are available, try to collect new data
        if not sentiment_data:
            logger.info("No existing sentiment data found. Checking API credentials...")
            
            # Initialize clients
            self._initialize_twitter()
            self._initialize_reddit()
            
            all_data = []
            
            # Keywords to search for
            keywords = ["FTSE", "S&P 500", "stock market", "economy", "inflation"]
            
            # Collect Twitter data if available
            if self.twitter_client:
                for keyword in tqdm(keywords, desc="Collecting Twitter data"):
                    tweets = self.get_tweets(keyword, limit=50)
                    if not tweets.empty:
                        all_data.append(tweets)
                    time.sleep(1)  # Rate limiting
            
            # Collect Reddit data if available
            if self.reddit_client:
                subreddits = ["investing", "stocks", "economy", "UKInvesting"]
                for subreddit in tqdm(subreddits, desc="Collecting Reddit data"):
                    for keyword in keywords[:2]:  # Limit keywords for Reddit
                        posts = self.get_reddit_posts(subreddit, keyword, limit=25)
                        if not posts.empty:
                            all_data.append(posts)
                        time.sleep(1)  # Rate limiting
            
            # Combine collected data
            if all_data:
                combined_data = pd.concat([df for df in all_data if not df.empty], ignore_index=True)
                analyzed_data = self.analyze_sentiment(combined_data)
                sentiment_data['social_media_sentiment'] = analyzed_data
                logger.info(f"Collected {len(analyzed_data)} sentiment records")
            else:
                logger.info("No API credentials available for sentiment data collection. Skipping sentiment analysis.")
        
        return sentiment_data

    def save_sentiment_data(self, sentiment_data: pd.DataFrame):
        """
        Save sentiment data to CSV files.
        
        Args:
            sentiment_data (pd.DataFrame): Sentiment data
        """
        from config import RAW_DATA_DIR
        
        filename = "sentiment_data.csv"
        filepath = RAW_DATA_DIR / filename
        sentiment_data.to_csv(filepath, index=False)
        logger.info(f"Sentiment data saved to {filepath}")

def main():
    """
    Main function to run sentiment data collection.
    """
    collector = SentimentDataCollector()
    
    # Collect all sentiment data
    sentiment_data = collector.collect_all_sentiment_data()
    
    if not sentiment_data.empty:
        # Save data
        collector.save_sentiment_data(sentiment_data)
        logger.info("Sentiment data collection completed.")
        
        # Print summary
        print(f"\nSentiment Data Collection Summary:")
        print(f"  Total records: {len(sentiment_data)}")
        if 'sentiment_score' in sentiment_data.columns:
            print(f"  Average sentiment: {sentiment_data['sentiment_score'].mean():.3f}")
            print(f"  Sentiment distribution: {sentiment_data['sentiment_category'].value_counts().to_dict()}")
    else:
        logger.error("No sentiment data collected. Please check your configuration and API access.")

if __name__ == "__main__":
    main() 