import logging
from logging import StreamHandler, FileHandler, Formatter, getLogger


def setup_logger(
    name,
    log_file=None,
    message_format="[%(asctime)s|%(levelname)s] - %(message)s",
    level=logging.INFO,
):
    logger = getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        formatter = Formatter(message_format, datefmt=r"%Y-%m-%d %H:%M:%S")

        if log_file:
            fh = FileHandler(log_file, mode="w", encoding="utf8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        ch = StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.propagate = False

    return logger
