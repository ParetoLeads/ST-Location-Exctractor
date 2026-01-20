"""
Retry handler utilities for API calls with exponential backoff.
"""
import time
import requests
from typing import Callable, TypeVar, Optional, Any
from functools import wraps
from config import config
from utils.exceptions import OSMQueryError

T = TypeVar('T')


def retry_with_backoff(
    max_attempts: Optional[int] = None,
    base_delay: Optional[int] = None,
    timeout: Optional[int] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (defaults to config value)
        base_delay: Base delay in seconds between retries (defaults to config value)
        timeout: Request timeout in seconds (defaults to config value)
        log_callback: Optional callback function for logging retry attempts
        
    Returns:
        Decorated function that will retry on failure
    """
    if max_attempts is None:
        max_attempts = config.MAX_RETRY_ATTEMPTS
    if base_delay is None:
        base_delay = config.RETRY_DELAY_BASE
    if timeout is None:
        timeout = config.OSM_API_TIMEOUT + config.OSM_API_TIMEOUT_BUFFER
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    wait_time = (attempt + 1) * base_delay
                    
                    if attempt < max_attempts - 1:
                        status_code = e.response.status_code if e.response else None
                        error_msg = _get_http_error_message(status_code, attempt + 1, max_attempts, wait_time)
                        
                        if log_callback:
                            log_callback(error_msg)
                        else:
                            print(error_msg)
                        
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = f"Failed after {max_attempts} attempts. HTTP error {status_code}"
                        if log_callback:
                            log_callback(error_msg)
                        raise OSMQueryError(error_msg) from e
                        
                except requests.exceptions.Timeout:
                    last_exception = TimeoutError("Request timeout")
                    wait_time = (attempt + 1) * base_delay
                    
                    if attempt < max_attempts - 1:
                        error_msg = (
                            f"Request timeout. Waiting {wait_time}s before retry "
                            f"(attempt {attempt + 1}/{max_attempts})..."
                        )
                        if log_callback:
                            log_callback(error_msg)
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = f"Request timeout after {max_attempts} attempts"
                        if log_callback:
                            log_callback(error_msg)
                        raise OSMQueryError(error_msg) from last_exception
                        
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    wait_time = (attempt + 1) * base_delay
                    
                    if attempt < max_attempts - 1:
                        error_msg = (
                            f"Request error: {str(e)}. Waiting {wait_time}s before retry "
                            f"(attempt {attempt + 1}/{max_attempts})..."
                        )
                        if log_callback:
                            log_callback(error_msg)
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = f"Request error after {max_attempts} attempts: {str(e)}"
                        if log_callback:
                            log_callback(error_msg)
                        raise OSMQueryError(error_msg) from e
                        
                except Exception as e:
                    last_exception = e
                    wait_time = (attempt + 1) * base_delay
                    
                    if attempt < max_attempts - 1:
                        error_msg = (
                            f"Unexpected error: {str(e)}. Waiting {wait_time}s before retry "
                            f"(attempt {attempt + 1}/{max_attempts})..."
                        )
                        if log_callback:
                            log_callback(error_msg)
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = f"Unexpected error after {max_attempts} attempts: {str(e)}"
                        if log_callback:
                            log_callback(error_msg)
                        raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise OSMQueryError(f"Failed after {max_attempts} attempts") from last_exception
            raise OSMQueryError(f"Failed after {max_attempts} attempts")
        
        return wrapper
    return decorator


def _get_http_error_message(
    status_code: Optional[int],
    attempt: int,
    max_attempts: int,
    wait_time: int
) -> str:
    """Generate user-friendly error message for HTTP errors."""
    if status_code == 429:  # Too Many Requests
        return (
            f"Rate limited (429). Waiting {wait_time}s before retry "
            f"(attempt {attempt}/{max_attempts})..."
        )
    elif status_code == 504:  # Gateway Timeout
        return (
            f"Gateway timeout (504). Waiting {wait_time}s before retry "
            f"(attempt {attempt}/{max_attempts})..."
        )
    else:
        return (
            f"HTTP error {status_code}. Waiting {wait_time}s before retry "
            f"(attempt {attempt}/{max_attempts})..."
        )


def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: Optional[int] = None,
    base_delay: Optional[int] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    **kwargs: Any
) -> T:
    """
    Execute a function with retry logic (non-decorator version).
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        log_callback: Optional callback function for logging
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
        
    Raises:
        OSMQueryError: If all retry attempts fail
    """
    if max_attempts is None:
        max_attempts = config.MAX_RETRY_ATTEMPTS
    if base_delay is None:
        base_delay = config.RETRY_DELAY_BASE
    
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            last_exception = e
            wait_time = (attempt + 1) * base_delay
            status_code = e.response.status_code if e.response else None
            
            if attempt < max_attempts - 1:
                error_msg = _get_http_error_message(status_code, attempt + 1, max_attempts, wait_time)
                if log_callback:
                    log_callback(error_msg)
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Failed after {max_attempts} attempts. HTTP error {status_code}"
                if log_callback:
                    log_callback(error_msg)
                raise OSMQueryError(error_msg) from e
                
        except requests.exceptions.Timeout:
            last_exception = TimeoutError("Request timeout")
            wait_time = (attempt + 1) * base_delay
            
            if attempt < max_attempts - 1:
                error_msg = (
                    f"Request timeout. Waiting {wait_time}s before retry "
                    f"(attempt {attempt + 1}/{max_attempts})..."
                )
                if log_callback:
                    log_callback(error_msg)
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Request timeout after {max_attempts} attempts"
                if log_callback:
                    log_callback(error_msg)
                raise OSMQueryError(error_msg) from last_exception
                
        except requests.exceptions.RequestException as e:
            last_exception = e
            wait_time = (attempt + 1) * base_delay
            
            if attempt < max_attempts - 1:
                error_msg = (
                    f"Request error: {str(e)}. Waiting {wait_time}s before retry "
                    f"(attempt {attempt + 1}/{max_attempts})..."
                )
                if log_callback:
                    log_callback(error_msg)
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Request error after {max_attempts} attempts: {str(e)}"
                if log_callback:
                    log_callback(error_msg)
                raise OSMQueryError(error_msg) from e
                
        except Exception as e:
            last_exception = e
            wait_time = (attempt + 1) * base_delay
            
            if attempt < max_attempts - 1:
                error_msg = (
                    f"Unexpected error: {str(e)}. Waiting {wait_time}s before retry "
                    f"(attempt {attempt + 1}/{max_attempts})..."
                )
                if log_callback:
                    log_callback(error_msg)
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Unexpected error after {max_attempts} attempts: {str(e)}"
                if log_callback:
                    log_callback(error_msg)
                raise
    
    # Should never reach here
    if last_exception:
        raise OSMQueryError(f"Failed after {max_attempts} attempts") from last_exception
    raise OSMQueryError(f"Failed after {max_attempts} attempts")
