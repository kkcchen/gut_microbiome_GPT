from accelerate.logging import get_logger
import logging
import sys

logger = get_logger('trainers')
base_logger = logger.logger

# check if logger has been initialized
if not base_logger.hasHandlers() or len(base_logger.handlers) == 0:
    base_logger.propagate = False
    base_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    base_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    base_logger.addHandler(handler)