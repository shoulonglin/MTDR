import logging


def get_logger():
    logger = logging.getLogger('mainlogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('./info.log', 'w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger
