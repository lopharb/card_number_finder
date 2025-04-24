import logging
from pathlib import Path


def get_logger(name: str = "app"):
    """
    Get a logger with the given name.

    Args:
        name (str): The name of the logger. Defaults to "app".

    Returns:
        logging.Logger: The logger with the given name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(
            Path(__file__).parent.parent / f"{name}.log", mode="a"
        )
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
