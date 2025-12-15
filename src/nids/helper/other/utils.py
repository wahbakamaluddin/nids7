import logging
import uuid
from itertools import islice, zip_longest

import numpy


def get_logger(debug=False):
    logger = logging.getLogger("cicflowmeter")
    if not logger.hasHandlers():
        logging.basicConfig()
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    return logger


def grouper(iterable, n, max_groups=0, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""

    if max_groups > 0:
        iterable = islice(iterable, max_groups * n)

    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def random_string():
    return uuid.uuid4().hex[:6].upper().replace("0", "X").replace("O", "Y")


def get_statistics(alist: list):
    """Get summary statistics of a list"""
    stats = dict()
    alist = [float(x) for x in alist]

    if len(alist) > 1:
        stats["total"] = sum(alist)
        stats["max"] = max(alist)
        stats["min"] = min(alist)
        stats["mean"] = numpy.mean(alist)
        stats["std"] = numpy.sqrt(numpy.var(alist))
    else:
        stats["total"] = 0
        stats["max"] = 0
        stats["min"] = 0
        stats["mean"] = 0
        stats["std"] = 0

    return stats
