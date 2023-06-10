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
from random import sample


class Solver:
    """
    This is the solver...
    """

    def __init__(self, model_parameters):
        self._mapping = {}
        self._model_parameters = model_parameters

    def run(self):
        policy = self.solve(self._model_parameters)
        board = environment.TicTacToe()
        while not board.get_state().is_over():
            opponent_moves = board.get_possible_moves(board.get_state())
            n_move = sample(opponent_moves, 1)
            board.mark(n_move,'X')
            board.get_state().print_state()
            if board.get_state().is_over():
                break
            agent_move = self.choose_action(board.get_state(),board.get_possible_moves(),policy)
            board.mark(agent_move, 'O')
            board.get_state().print_state()

    def solve(self, model_parameters):
        board = environment.TicTacToe()
        Q = {}
        for state in board.get_states().values():
            for action in board.get_possible_moves(state):
                Q[(str(state.BOARD), action)] = None
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
            if (str(_state.BOARD), _action) not in Q:
                raise Exception(f"State: {_state.BOARD}, Action: {_action} not in Q")
            return Q[(str(_state.BOARD), _action)]

        def __update_Q_value(_state, _action, _reward, _next_state, _board):
            next_Q_values = [__get_Q_value(_next_state, next_action) for next_action in
                             _board.get_possible_moves(_next_state)]
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            if (str(_state.BOARD), _action) in Q:
                Q[(str(_state.BOARD), _action)] += alpha * (
                        _reward + discount_factor * max_next_Q - Q[(str(_state.BOARD), _action)])
            else:
                raise Exception(f"State: {_state.BOARD} has no action {_action}")

        for i in tqdm(range(iterations), desc='Q-Learning'):
            board.reset()
            while not board.get_state().is_over():
                if i % 2 == 0:
                    mark = 'X'
                else:
                    mark = 'O'
                state = copy.deepcopy(board.get_state())
                available_moves = board.get_possible_moves(state)
                action = __choose_action(state, available_moves)
                next_state, reward = board.mark(action, mark)
                __update_Q_value(state, action, reward, next_state, board)
                board.update_state(next_state)
        with open("Q_values.txt", "w") as qd:
            list_of_strings = [f'{key[0]}, {key[1]}: {Q[key]}' for key in Q.keys() if Q[key] is not None]
            [qd.write(f'{st}\n') for st in list_of_strings]
        return Q

    def get_policy(self):
        return self._mapping

    def choose_action(self, state, possible_moves, policy):
        Q_values = [policy(state, _action) for _action in possible_moves]
        max_Q = max(Q_values)
        if Q_values.count(max_Q) > 1:
            best_moves = [i for i in range(len(possible_moves)) if Q_values[i] == max_Q]
            i = random.choice(best_moves)
        else:
            i = Q_values.index(max_Q)
        return possible_moves[i]

