from jax import random
DEBUG = False
SPARSE = False
Board_State = None
aiAgent = None
Opponent = None
TicTacToe = None
ORIGINAL_STATES = None
REAL_MODEL_PARAMETERS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
MAX = float('+inf')
MIN = float('-inf')
ITERATIONS = 10000
GAMES_TEST = 1000
WIN_REWARD = 10
LOSE_REWARD = -10
DRAW_REWARD = -5
DISCOUNT_FACTOR = 0.95
ALPHA = 0.5
EPSILON = 0.3
THETA = 1e-5
IMMEDIATE_REWARD = -1
SEED = random.PRNGKey(0)

