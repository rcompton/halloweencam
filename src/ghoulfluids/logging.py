import logging
import sys


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """
    Configures the root logger for the application.

    Args:
        log_level: The minimum log level to output (e.g., "INFO", "DEBUG").
        log_file: If provided, logs will be written to this file. Otherwise, logs
            will be written to stdout.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs and allow reconfiguration
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for the given name.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A logger instance.
    """
    return logging.getLogger(name)
