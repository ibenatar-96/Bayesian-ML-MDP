import logging

"""
Creates Observation File if it does not exist,
"""


class Loggers:
    def __init__(self):
        self.observations = self.init_logger('observations', logging.INFO)
        self.games_won = self.init_logger('games_won_over_time', logging.INFO)

    @staticmethod
    def init_logger(log_name, level):
        logging.basicConfig(filename=f"{log_name}.log", encoding='utf-8', level=level)
        return logging.getLogger(log_name)
