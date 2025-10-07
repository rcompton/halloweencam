import logging
import sys

def setup_logging(log_level: str = "INFO"):
    """
    Configures the root logger for the application.

    Args:
        log_level: The minimum log level to output (e.g., "INFO", "DEBUG").
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for the given name.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
