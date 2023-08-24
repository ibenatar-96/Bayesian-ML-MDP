"""
This file defines global variables.
"""
import os
import actions

DEBUG = False
DEBUG_BOARD = False
PLOT = True
SPARSE = False
CLEAN_FILES = True
INIT_OBSERVATIONS = True
aiAgent = None
Opponent = None
ORIGINAL_STATES = None
REAL_MODEL_PARAMETERS = {('ai_mark', 1): 1.0, ('ai_mark', 2): 1.0, ('ai_mark', 3): 1.0, ('ai_mark', 4): 1.0,
                         ('ai_mark', 5): 0.3, ('ai_mark', 6): 1.0, ('ai_mark', 7): 1.0, ('ai_mark', 8): 1.0,
                         ('ai_mark', 9): 1.0}
MAX = float('+inf')
MIN = float('-inf')
ITERATIONS = 50
GAMES_TEST = int(ITERATIONS / 4)
GAMES_COLLECT = 15
WIN_REWARD = 10
LOSE_REWARD = -10
DRAW_REWARD = 0
DISCOUNT_FACTOR = 0.95
ALPHA = 0.5
EPSILON = 0.2
THETA = 1e-5
IMMEDIATE_REWARD = -1
LOG_FILE = "observations.log"
GAMES_WIN_RATIO_FILE = "games_won_over_time.log"
IGNORE_ACTIONS = ['opponent_mark']
INIT_MODEL_PARAMETERS = {('ai_mark', 5): 0.8}
GAMES_WON_OVER_TIME = []


def largest_divisors(x):
    largest_divisor1 = 1
    largest_divisor2 = x
    for i in range(2, int(x ** 0.5) + 1):
        if x % i == 0:
            largest_divisor1 = i
            largest_divisor2 = x // i

    return largest_divisor2, largest_divisor1


def init_model_params():
    """
    Also initializes the utils.INIT_MODEL_PARAMETERS, which is the initial belief of the probability for each action.
    an action is composed of a tuple (function name, action parameter),
    For example: ('ai_mark', 2) means,
    The action 'ai_mark' - it is the AI models' turn to mark, and it marks the 2nd cell.
    utils.INIT_MODEL_PARAMETERS = {
    ('ai_mark', 1): 1.0
    ('ai_mark', 2): 1.0
    ...
    ('ai_mark', 5): 0.8
    ...}
    """
    for (func_action, action_param) in actions.Actions().get_actions():
        if func_action not in IGNORE_ACTIONS and (func_action, action_param) not in INIT_MODEL_PARAMETERS:
            INIT_MODEL_PARAMETERS[(func_action, action_param)] = 1.0
