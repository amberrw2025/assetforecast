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
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from utils.logger import get_logger

logger = get_logger("sentiment_data")

class SentimentDataCollector:
    """
    Collects sentiment data from social media platforms and news sources.
    """
    
    def __init__(self):
        self.start_date = "2023-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
    def get_twitter_sentiment(self, keywords: List[str], max_tweets: int = 1000) -> pd.DataFrame:
        """
        Fetch Twitter sentiment data for given keywords.
        
        Args:
            keywords (List[str]): Keywords to search for
            max_tweets (int): Maximum number of tweets to fetch per keyword
            
        Returns:
            pd.DataFrame: Twitter sentiment data
        """
        try:
            if not all([TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET]):
                logger.warning("Twitter API credentials not provided. Using sample data.")
                return self._generate_sample_twitter_data(keywords)
            
            import tweepy
            
            # Authenticate with Twitter
            auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
            auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
            api = tweepy.API(auth, wait_on_rate_limit=True)
            
            all_tweets = []
            
            for keyword in tqdm(keywords, desc="Fetching Twitter data"):
                try:
                    # Search tweets
                    tweets = tweepy.Cursor(
                        api.search_tweets,
                        q=keyword,
                        lang="en",
                        tweet_mode="extended",
                        since_id=None
                    ).items(max_tweets)
                    
                    keyword_tweets = []
                    for tweet in tweets:
                        tweet_data = {
                            'date': tweet.created_at,
                            'text': tweet.full_text,
                            'user_followers': tweet.user.followers_count,
                            'retweet_count': tweet.retweet_count,
                            'favorite_count': tweet.favorite_count,
                            'keyword': keyword,
                            'source': 'Twitter'
                        }
                        keyword_tweets.append(tweet_data)
                    
                    all_tweets.extend(keyword_tweets)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching Twitter data for {keyword}: {str(e)}")
                    continue
            
            if all_tweets:
                df = pd.DataFrame(all_tweets)
                logger.info(f"Successfully fetched {len(df)} tweets")
                return df
            else:
                return pd.DataFrame()
                
        except ImportError:
            logger.error("tweepy not installed. Please install with: pip install tweepy")
            return self._generate_sample_twitter_data(keywords)
        except Exception as e:
            logger.error(f"Error in Twitter data collection: {str(e)}")
            return self._generate_sample_twitter_data(keywords)
    
    def _generate_sample_twitter_data(self, keywords: List[str]) -> pd.DataFrame:
        """
        Generate sample Twitter data for testing purposes.
        
        Args:
            keywords (List[str]): Keywords to generate data for
            
        Returns:
            pd.DataFrame: Sample Twitter data
        """
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        all_tweets = []
        
        for keyword in keywords:
            # Generate 10-50 tweets per keyword
            num_tweets = np.random.randint(10, 50)
            
            for _ in range(num_tweets):
                # Random date
                tweet_date = np.random.choice(dates)
                
                # Sample tweet texts
                sample_texts = [
                    f"Great news about {keyword}!",
                    f"Concerned about {keyword} trends",
                    f"{keyword} showing positive momentum",
                    f"Market analysis on {keyword}",
                    f"Interesting developments in {keyword}"
                ]
                
                tweet_data = {
                    'date': tweet_date,
                    'text': np.random.choice(sample_texts),
                    'user_followers': np.random.randint(100, 10000),
                    'retweet_count': np.random.randint(0, 100),
                    'favorite_count': np.random.randint(0, 500),
                    'keyword': keyword,
                    'source': 'Twitter (Sample)'
                }
                all_tweets.append(tweet_data)
        
        df = pd.DataFrame(all_tweets)
        logger.info(f"Generated sample Twitter data for {len(keywords)} keywords")
        return df
    
    def get_reddit_sentiment(self, subreddits: List[str], keywords: List[str], max_posts: int = 500) -> pd.DataFrame:
        """
        Fetch Reddit sentiment data for given subreddits and keywords.
        
        Args:
            subreddits (List[str]): Subreddits to search in
            keywords (List[str]): Keywords to search for
            max_posts (int): Maximum number of posts to fetch per subreddit
            
        Returns:
            pd.DataFrame: Reddit sentiment data
        """
        try:
            if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
                logger.warning("Reddit API credentials not provided. Using sample data.")
                return self._generate_sample_reddit_data(subreddits, keywords)
            
            import praw
            
            # Initialize Reddit client
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            
            all_posts = []
            
            for subreddit_name in tqdm(subreddits, desc="Fetching Reddit data"):
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Search for posts containing keywords
                    for keyword in keywords:
                        search_query = f"{keyword}"
                        posts = subreddit.search(search_query, limit=max_posts//len(keywords))
                        
                        for post in posts:
                            post_data = {
                                'date': datetime.fromtimestamp(post.created_utc),
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'subreddit': subreddit_name,
                                'keyword': keyword,
                                'source': 'Reddit'
                            }
                            all_posts.append(post_data)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching Reddit data for r/{subreddit_name}: {str(e)}")
                    continue
            
            if all_posts:
                df = pd.DataFrame(all_posts)
                logger.info(f"Successfully fetched {len(df)} Reddit posts")
                return df
            else:
                return pd.DataFrame()
                
        except ImportError:
            logger.error("praw not installed. Please install with: pip install praw")
            return self._generate_sample_reddit_data(subreddits, keywords)
        except Exception as e:
            logger.error(f"Error in Reddit data collection: {str(e)}")
            return self._generate_sample_reddit_data(subreddits, keywords)
    
    def _generate_sample_reddit_data(self, subreddits: List[str], keywords: List[str]) -> pd.DataFrame:
        """
        Generate sample Reddit data for testing purposes.
        
        Args:
            subreddits (List[str]): Subreddits to generate data for
            keywords (List[str]): Keywords to generate data for
            
        Returns:
            pd.DataFrame: Sample Reddit data
        """
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        all_posts = []
        
        for subreddit in subreddits:
            for keyword in keywords:
                # Generate 5-20 posts per subreddit-keyword combination
                num_posts = np.random.randint(5, 20)
                
                for _ in range(num_posts):
                    # Random date
                    post_date = np.random.choice(dates)
                    
                    # Sample post titles
                    sample_titles = [
                        f"Discussion: {keyword} trends",
                        f"Analysis of {keyword} performance",
                        f"Thoughts on {keyword}?",
                        f"{keyword} - what's your take?",
                        f"Market update: {keyword}"
                    ]
                    
                    # Sample post texts
                    sample_texts = [
                        f"Looking at the recent {keyword} data...",
                        f"What do you think about {keyword}?",
                        f"Interesting developments in {keyword}.",
                        f"Analysis of {keyword} market conditions.",
                        f"Your thoughts on {keyword}?"
                    ]
                    
                    post_data = {
                        'date': post_date,
                        'title': np.random.choice(sample_titles),
                        'text': np.random.choice(sample_texts),
                        'score': np.random.randint(-10, 100),
                        'upvote_ratio': np.random.uniform(0.5, 1.0),
                        'num_comments': np.random.randint(0, 50),
                        'subreddit': subreddit,
                        'keyword': keyword,
                        'source': 'Reddit (Sample)'
                    }
                    all_posts.append(post_data)
        
        df = pd.DataFrame(all_posts)
        logger.info(f"Generated sample Reddit data for {len(subreddits)} subreddits and {len(keywords)} keywords")
        return df
    
    def calculate_sentiment_scores(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Calculate sentiment scores for text data.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            
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
            df['sentiment_score'] = df[text_column].apply(get_sentiment)
            
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
        Collect all sentiment data from various sources.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of sentiment data by source
        """
        sentiment_data = {}
        
        # Define keywords for sentiment analysis
        keywords = ['inflation', 'recession', 'interest rates', 'oil prices', 'unemployment', 
                   'stock market', 'economy', 'GDP', 'employment']
        
        # Collect Twitter sentiment data
        logger.info("Collecting Twitter sentiment data...")
        twitter_data = self.get_twitter_sentiment(keywords)
        
        if not twitter_data.empty:
            twitter_data = self.calculate_sentiment_scores(twitter_data, 'text')
            sentiment_data['twitter'] = twitter_data
        
        # Collect Reddit sentiment data
        logger.info("Collecting Reddit sentiment data...")
        subreddits = ['investing', 'stocks', 'economics', 'finance', 'wallstreetbets']
        reddit_data = self.get_reddit_sentiment(subreddits, keywords)
        
        if not reddit_data.empty:
            reddit_data = self.calculate_sentiment_scores(reddit_data, 'text')
            sentiment_data['reddit'] = reddit_data
        
        logger.info(f"Sentiment data collection completed. Sources: {list(sentiment_data.keys())}")
        return sentiment_data
    
    def save_sentiment_data(self, sentiment_data: Dict[str, pd.DataFrame]):
        """
        Save sentiment data to CSV files.
        
        Args:
            sentiment_data (Dict[str, pd.DataFrame]): Sentiment data by source
        """
        from config import RAW_DATA_DIR
        
        saved_files = []
        
        for source, data in sentiment_data.items():
            if not data.empty:
                filename = f"sentiment_{source}.csv"
                filepath = RAW_DATA_DIR / filename
                data.to_csv(filepath, index=False)
                saved_files.append(filepath)
                logger.info(f"Sentiment data saved to {filepath}")
        
        return saved_files

def main():
    """
    Main function to run sentiment data collection.
    """
    collector = SentimentDataCollector()
    
    # Collect all sentiment data
    sentiment_data = collector.collect_all_sentiment_data()
    
    if sentiment_data:
        # Save data
        saved_files = collector.save_sentiment_data(sentiment_data)
        logger.info(f"Sentiment data collection completed. Files saved: {saved_files}")
        
        # Print summary
        print(f"\nSentiment Data Collection Summary:")
        for source, data in sentiment_data.items():
            print(f"{source.upper()}: {len(data)} records")
            if 'sentiment_score' in data.columns:
                print(f"  Average sentiment: {data['sentiment_score'].mean():.3f}")
                print(f"  Sentiment distribution: {data['sentiment_category'].value_counts().to_dict()}")
    else:
        logger.error("No sentiment data collected. Please check your configuration and API access.")

if __name__ == "__main__":
    main() 