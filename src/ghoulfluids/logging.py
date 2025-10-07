import logging
import sys
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configures the root logger for the application.

    Args:
        log_level: The minimum log level to output (e.g., "INFO", "DEBUG").
        log_file: If provided, log to this file instead of stdout.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a new handler
    if log_file:
        handler = logging.FileHandler(log_file, mode="a")
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for the given name.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
