"""
Data acquisition modules for the Forecast Accuracy Assessment Model Pipeline.
"""

from .financial_data import FinancialDataCollector
from .economic_data import EconomicDataCollector
from .sentiment_data import SentimentDataCollector

__all__ = [
    'FinancialDataCollector',
    'EconomicDataCollector', 
    'SentimentDataCollector'
] 