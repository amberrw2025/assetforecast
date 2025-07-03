"""
Utility decorators for the Forecast Accuracy Assessment Model.
"""

import time
import random
from functools import wraps
from .logger import get_logger

logger = get_logger("decorators")

def retry(exceptions, tries=4, delay=3, backoff=2, jitter=1):
    """
    Retry calling the decorated function using an exponential backoff.

    Args:
        exceptions (Exception or tuple): The exception to check. May be a tuple of
            exceptions to check.
        tries (int): Number of times to try (not retry) before giving up.
        delay (int): Initial delay between retries in seconds.
        backoff (int): Backoff multiplier (e.g. value of 2 will double the delay
            each retry).
        jitter (int): Adds a random jitter to the delay time.
    """
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    logger.warning(msg)
                    time.sleep(mdelay + random.uniform(0, jitter))
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry 