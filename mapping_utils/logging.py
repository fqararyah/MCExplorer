import __init__
import logging
from functools import wraps
from constants import ENABLE_LOGGING, LOGGING_PATH
import os

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers:
    logger.handlers.clear()

log_file = LOGGING_PATH + '/my_log_file.log'
max_bytes = 100 * 1024  # 100 KB
backup_count = 0        # Keep no backup files

# Custom handler to overwrite the log file when it exceeds max size
class FileSizeRotatingHandler(logging.Handler):
    def __init__(self, check_interval=100):
        super().__init__()
        self.check_interval = check_interval  # Number of logs before checking file size
        self.log_count = 0  # Counter to track the number of logs

    def emit(self, record):
        # Every 'check_interval' logs, check the file size
        self.log_count += 1
        if self.log_count >= self.check_interval:
            self.log_count = 0  # Reset counter
            # Check if the file size exceeds the limit
            if os.path.exists(log_file) and os.path.getsize(log_file) > max_bytes:
                with open(log_file, 'w'): pass  # Clear the file
        
        # Now log the message by writing it to the file
        try:
            log_entry = self.format(record)  # Format the log record
            with open(log_file, 'a') as f:
                f.write(log_entry + '\n')  # Write the log to the file
        except Exception:
            self.handleError(record)

# Set up custom file handler
file_handler = FileSizeRotatingHandler()
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Add handler to logger
logger.addHandler(file_handler)

# Add handlers
logger.addHandler(file_handler)

def log_function_call(func):
    if not ENABLE_LOGGING:
        # Return the function unwrapped if logging is disabled
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.exception(f"{func.__name__} raised an exception: {e}")
            raise

    return wrapper
