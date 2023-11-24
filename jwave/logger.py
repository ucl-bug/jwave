import logging

# Initialize the logger
logger = logging.getLogger(__name__.split(".")[0])
logger.setLevel(logging.INFO)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


# Function to set logging level
def set_logging_level(level: int) -> None:
    """
    Set the logging level for both the logger and all its handlers.

    This function updates the logging level of the logger to the specified
    level and also iterates through all the handlers associated with the logger,
    updating their logging levels to match the specified level.

    Parameters:
    level (int): An integer representing the logging level. This should be one
                 of the logging level constants defined in the logging module, such as
                 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, or logging.CRITICAL.

    Returns:
    None
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
