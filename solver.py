import os.path

import runtime
import environment
import copy
import numpyro
import numpyro.distributions as dist
import jax
import time
import random
from tqdm import tqdm
import os


class Solver:
    """
    This is the solver...
    """

    def __init__(self, model_parameters):
        self._mapping = {}
        self._model_parameters = model_parameters

    def run(self):
        if os.path.exists("Q_values.txt") and os.path.isfile("Q_values.txt"):
            # TODO: load Q_values into policy..
            pass
        policy = self.solve(self._model_parameters)
        while not runtime.Board_State.is_over():
            break

    def solve(self, model_parameters):
        Q = {}
        for state in runtime.TicTacToe.get_states().values():
            for action in runtime.TicTacToe.get_possible_moves(state):
                Q[(state, action)] = 0.0
        alpha = runtime.ALPHA
        epsilon = runtime.EPSILON
        discount_factor = runtime.DISCOUNT_FACTOR
        iterations = runtime.ITERATIONS

        def __choose_action(_state, _available_moves):
            if random.uniform(0, 1) < epsilon:
                return random.choice(_available_moves)
            else:
                Q_values = [__get_Q_value(_state, _action) for _action in _available_moves]
                max_Q = max(Q_values)
                if Q_values.count(max_Q) > 1:
                    best_moves = [i for i in range(len(_available_moves)) if Q_values[i] == max_Q]
                    i = random.choice(best_moves)
                else:
                    i = Q_values.index(max_Q)
            return _available_moves[i]

        def __get_Q_value(_state, _action):
            if (_state, _action) not in Q:
                Q[(_state, _action)] = 0.0
            return Q[(_state, _action)]

        def __update_Q_value(_state, _action, _reward, _next_state, _board):
            next_Q_values = [__get_Q_value(_next_state, next_action) for next_action in
                             _board.get_possible_moves(_next_state)]
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            if (_state, _action) in Q:
                Q[(_state, _action)] += alpha * (
                        _reward + discount_factor * max_next_Q - Q[(_state, _action)])
            else:
                print(f"State: {_state.BOARD} has no action {_action}")
                raise Exception("why..")

        for _ in tqdm(range(iterations), desc='tqdm() Progress Bar'):
            board = environment.TicTacToe()
            while not board.get_state().is_over():
                available_moves = board.get_possible_moves(board.get_state())
                action = __choose_action(board.get_state(), available_moves)
                prev_state = board.get_state()
                next_state, reward = board.mark(action, 'O')
                __update_Q_value(prev_state, action, reward, next_state, board)
                board.update_state(next_state)
        with open("Q_values.txt", "w") as qd:
            list_of_strings = [f'{key[0].BOARD} : {key[1]}' for key in Q.keys()]
            [qd.write(f'{st}\n') for st in list_of_strings]
        return Q

    def get_policy(self):
        return self._mapping
