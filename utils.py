import os
"""
This file defines global variables.
"""
DEBUG = False
DEBUG_BOARD = False
PLOT = False
SPARSE = False
CLEAN_FILES = True
INIT_OBSERVATIONS = True
aiAgent = None
Opponent = None
ORIGINAL_STATES = None
REAL_MODEL_PARAMETERS = {('ai_mark', 1): 1.0, ('ai_mark', 2): 1.0, ('ai_mark', 3): 1.0, ('ai_mark', 4): 1.0,
                         ('ai_mark', 5): 1.0, ('ai_mark', 6): 1.0, ('ai_mark', 7): 1.0, ('ai_mark', 8): 1.0,
                         ('ai_mark', 9): 1.0}
MAX = float('+inf')
MIN = float('-inf')
ITERATIONS = 25000
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
INIT_OBSERVATIONS_LEN = 10
LOG_FILE = os.path.join("logs", "observations.log")
GAMES_WIN_RATIO_FILE = os.path.join("logs", "games_win_ratio.log")
IGNORE_ACTIONS = ['opponent_mark']
INIT_MODEL_PARAMETERS = {('ai_mark', 5): 1.0}
