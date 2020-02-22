import logging
import logging.handlers
import datetime
import os
import re

# from main.service.config.config import LOG_DIR

LOG_DIR = '/home/user/tmp/logs/api/asr'
LOG_FILE = LOG_DIR + '/info.log'


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# python 3 style
class MyLogger(object, metaclass=SingletonType):
    # __metaclass__ = SingletonType   # python 2 Style
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger("asr2text")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')

        now = datetime.datetime.now()
        dirname = LOG_DIR

        os.makedirs(dirname, exist_ok=True)

        # fileHandler = logging.FileHandler(dirname + "/info_" + now.strftime("%Y-%m-%d") + ".log")
        # fileHandler = logging.FileHandler(dirname + "/info.log")

        streamHandler = logging.StreamHandler()

        # fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        # self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

        print("Generate new instance")

    def get_logger(self):
        return self._logger
