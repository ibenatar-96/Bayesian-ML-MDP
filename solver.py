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
import json


class Solver:
    """
    This is the solver...
    """

    def __init__(self, model_parameters):
        self._mapping = {}
        self._model_parameters = model_parameters

    def run(self):
        if os.path.exists("Q_values.txt") and runtime.SPARSE:
            print("Q_values exists, loading..")
            policy = self._load_policy("Q_values.txt")
        else:
            policy = self.solve(self._model_parameters)
        board = environment.TicTacToe()
        while not board.get_state().is_over():
            opponent_moves = board.get_possible_moves(board.get_state())
            n_move = sample(opponent_moves, 1)[0]
            board.mark(n_move, 'X')
            board.get_state().print_state()
            if board.get_state().is_over():
                break
            agent_move = self.choose_action(board.get_state(), board.get_possible_moves(board.get_state()), policy)
            board.mark(agent_move, 'O')
            board.get_state().print_state()

    def solve(self, model_parameters):
        board = environment.TicTacToe()
        Q = {}
        for state in board.get_states().values():
            for action in board.get_possible_moves(state):
                Q[(str(state.BOARD), action)] = 0.0
        alpha = runtime.ALPHA
        epsilon = runtime.EPSILON
        discount_factor = runtime.DISCOUNT_FACTOR
        iterations = runtime.ITERATIONS

        def __choose_action(_state, _available_moves, _mark):
            if random.uniform(0, 1) < epsilon:
                return random.choice(_available_moves)
            else:
                Q_values = [__get_Q_value(_state, _action) for _action in _available_moves]
                if _mark == "O":
                    max_Q = max(Q_values)
                else:
                    max_Q = min(Q_values)
                if Q_values.count(max_Q) > 1:
                    best_moves = [i for i in range(len(_available_moves)) if Q_values[i] == max_Q]
                    i = random.choice(best_moves)
                else:
                    i = Q_values.index(max_Q)
            return _available_moves[i]

        def __get_Q_value(_state, _action):
            if (str(_state.BOARD), _action) not in Q:
                raise Exception(f"State: {_state.BOARD}, Action: {_action} not in Q")
            elif (str(_state.BOARD), _action) in Q and Q[(str(_state.BOARD), _action)] is None:
                Q[(str(_state.BOARD), _action)] = 0.0
            return Q[(str(_state.BOARD), _action)]

        def __update_Q_value(_state, _action, _reward, _next_state, _board):
            next_Q_values = [__get_Q_value(_next_state, next_action) for next_action in
                             _board.get_possible_moves(_next_state)]
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            if (str(_state.BOARD), _action) in Q:
                Q[(str(_state.BOARD), _action)] += alpha * (
                        _reward + discount_factor * max_next_Q - Q[(str(_state.BOARD), _action)])

                # if str(_state.BOARD) == [['O', None, 'X'], [None, None, 'X'], ['O', 'X', 'O']]:
                #     print(f"Q[{str(_state.BOARD)},{_action}] = {Q[(str(_state.BOARD), _action)]}")

            else:
                raise Exception(f"State: {_state.BOARD} has no action {_action}")

        for _ in tqdm(range(iterations), desc='Agent Q-Learning'):
            board.reset()
            i = 0
            while not board.get_state().is_over():
                played_twice = False
                if i % 2 == 0:
                    mark = 'X'
                    second_mark = 'O'
                else:
                    mark = 'O'
                    second_mark = 'X'
                state = copy.deepcopy(board.get_state())
                available_moves = board.get_possible_moves(state)
                action = __choose_action(state, available_moves, mark)
                next_state, reward = board.mark(action, mark)
                if not next_state.is_over():
                    state_ = copy.deepcopy(next_state)
                    available_moves_ = board.get_possible_moves(state_)
                    action_ = __choose_action(state_, available_moves_, second_mark)
                    next_state_, reward = board.mark(action_, second_mark)
                    played_twice = True
                if state.BOARD == [[None, None, 'X'], [None, 'O', None], ['X', 'O', 'X']]:
                    print(f"Next state: {next_state.BOARD}, reward: {reward}, mark: {mark}")
                # if mark == 'X' and played_twice:
                #     __update_Q_value(next_state, action_, reward, next_state_, board)
                __update_Q_value(state, action, reward, next_state, board)
                board.update_state(next_state)
                i += 1
        self._save_policy(Q, "Q_values.txt")
        return Q

    def get_policy(self):
        return self._mapping

    def choose_action(self, state, possible_moves, policy):
        Q_values = [policy[str(state.BOARD), _action] for _action in possible_moves]
        max_Q = max(Q_values)
        if Q_values.count(max_Q) > 1:
            best_moves = [i for i in range(len(possible_moves)) if Q_values[i] == max_Q]
            i = random.choice(best_moves)
        else:
            i = Q_values.index(max_Q)
        return possible_moves[i]

    def _save_policy(self, Q, txt_file):
        # for key, value in list(policy.items()):
        #     if value is None:
        #         del policy[key]
        jsonifable = {f"{str(state)}:{str(action)}": val for (state, action), val in Q.items()}
        with open(txt_file, "w") as qd:
            json.dump(jsonifable, qd)

        with open("Q_values_debug.txt", "w") as f:
            list_of_strings = [f'{key[0]}, {key[1]}: {Q[key]}' for key in Q.keys() if Q[key] is not None]
            [f.write(f'{st}\n') for st in list_of_strings]

    def _load_policy(self, txt_file):
        with open(txt_file, "r") as f:
            policy = json.load(f)
        d = {}
        for key, val in policy.items():
            start, end = key.split(':')
            d[str(start), int(end)] = val
        return d
