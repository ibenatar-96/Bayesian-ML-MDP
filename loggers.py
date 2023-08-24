import logging
import os

logging.getLogger('jax._src.xla_bridge').addFilter(lambda _: False)

"""
Creates Observation File if it does not exist,
"""

EPISODE_LEVEL = 9


class EpisodeLogger:
    def __init__(self, logger):
        self.logger = logger

    def episode(self, message, *args, **kwargs):
        if self.logger.isEnabledFor(EPISODE_LEVEL):
            self.logger._log(EPISODE_LEVEL, message, args, **kwargs)


class Loggers:
    def __init__(self):
        self.observations = self.init_logger('observations', format='%(levelname)s:%(message)s', level=logging.INFO, special_type='episode')
        self.games_won = self.init_logger('games_won_over_time', format='%(levelname)s:%(message)s',level=logging.INFO)

    def init_logger(self, log_name, format, level, special_type=None):
        if os.path.isfile(f"{log_name}.log"):
            with open(f"{log_name}.log", 'w') as log_file:
                log_file.truncate(0)
        logging.basicConfig(filename=f"{log_name}.log", format=format, encoding='utf-8', level=level)
        match special_type:
            case 'episode':
                logging.addLevelName(EPISODE_LEVEL, "EPISODE")
                logger = logging.getLogger(log_name)
                logger.setLevel(EPISODE_LEVEL)
                return EpisodeLogger(logger)
        return logging.getLogger(log_name)
