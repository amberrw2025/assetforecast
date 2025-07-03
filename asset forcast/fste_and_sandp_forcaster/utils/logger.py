"""
Logging utility for the Forecast Accuracy Assessment Model Pipeline.
Provides centralized logging configuration and utilities.
"""

import sys
from pathlib import Path
from loguru import logger
from config import LOGGING_CONFIG

def setup_logger():
    """
    Configure and setup the logger for the pipeline.
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        format=LOGGING_CONFIG["format"],
        level=LOGGING_CONFIG["level"],
        colorize=True
    )
    
    # Add file logger
    logger.add(
        LOGGING_CONFIG["file"],
        format=LOGGING_CONFIG["format"],
        level=LOGGING_CONFIG["level"],
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger

def get_logger(name: str = None):
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Name for the logger
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger

# Initialize logger
setup_logger() 