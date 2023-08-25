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
    def __init__(self, observations, games_won):
        self.observations = self.init_logger(observations, format='%(levelname)s:%(message)s', level=logging.INFO, special_type='episode')
        self.games_won = self.init_logger(games_won, format='%(levelname)s:%(message)s',level=logging.INFO)

    def init_logger(self, log_name, format, level, special_type=None):
        if os.path.isfile(log_name):
            with open(log_name, 'w') as log_file:
                log_file.truncate(0)
        logging.basicConfig(filename=log_name, format=format, encoding='utf-8', level=level)
        match special_type:
            case 'episode':
                logging.addLevelName(EPISODE_LEVEL, "EPISODE")
                logger = logging.getLogger(log_name)
                logger.setLevel(EPISODE_LEVEL)
                return EpisodeLogger(logger)
        return logging.getLogger(os.path.splitext(log_name)[0])
