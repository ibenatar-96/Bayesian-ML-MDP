import copy
import time
from functools import reduce
import runtime
from itertools import product
import numpyro
import numpyro.distributions as dist
import jax
from itertools import combinations
import ast
import os

# numpyro.set_platform("cpu")


class Environment:
    """
    Tic Tac Toe board with probabilities to mark in each square
    3x3 Matrix, each sqaure: 'X', 'O', None
    2 Players - AI Agent, and Human
    AI Agent - 'O'; chooses next square depending on policy from Q-Learning
    Human - 'X'; chooses next square with Uniform Prob.
    Terminal State = Winner (3 in a row / column / diagonal) or Full Board (no empty square) and Draw
    """

    def __init__(self, real_model_parameters):
        self._real_model_parameters = real_model_parameters
        # TODO: self._real_model_parameters = real.. this is mapping with REAL numbers! not prob. dist.
        self._states = self._init_states_space()
        self._state = self._states[str([[None] * 3 for _ in range(3)])]

    def __iter__(self):
        return iter(self._states)

    def __str__(self):
        output = ""
        for state in self._states:
            output += str(state) + "\n"
        return output

    def _init_states_space(self):
        start_time = time.time()
        states = {}
        square_mark = ['X', 'O', None]
        for row1 in product(square_mark, repeat=3):
            for row2 in product(square_mark, repeat=3):
                for row3 in product(square_mark, repeat=3):
                    state = [list(row1), list(row2), list(row3)]
                    term, win = self._is_terminal(state)
                    states[str(state)] = State(term, win, state)
        end_time = time.time()
        if runtime.DEBUG:
            print(f"Time for Creating State Space: {end_time - start_time}")
        return states

    @staticmethod
    def _is_terminal(board):
        for row in board:
            if all(i == row[0] for i in row) and row[0] is not None:
                return True, row[0]
        for j in range(3):
            column = [row[j] for row in board]
            if all(i == column[0] for i in column) and column[0] is not None:
                return True, column[0]
        if ((board[0][0] == board[1][1] == board[2][2]) or
            (board[0][2] == board[1][1] == board[2][0])) and \
                board[1][1] is not None:
            return True, board[1][1]
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    return False, None
        return True, None

    def get_state(self):
        return self._state

    def get_states(self):
        return self._states

    @staticmethod
    def get_possible_moves(state=None):
        _state = state
        if _state is None:
            _state = runtime.Board_State.BOARD
        if isinstance(_state, State):
            _state = _state.BOARD
        if not isinstance(_state, list):
            _state = eval(_state)

        moves = [(i * 3 + j + 1) for i in range(3) for j in range(3) if _state[i][j] is None]
        return moves

    def mark(self, next_move, mark, state=None):
        reward = -1
        if state is None:
            state = self._state.BOARD
        if not isinstance(state, list):
            state = eval(state)
        i = (next_move - 1) // 3
        j = (next_move - 1) % 3
        assert i * 3 + j + 1 == next_move
        assert (state[i][j] is None)
        y = numpyro.sample('y', dist.Bernoulli(probs=self._real_model_parameters[next_move]), rng_key=jax.random.PRNGKey(int(time.time() * 1E6))).item()
        if y > 0 or mark == 'X':
            state[i][j] = mark
            self._state = self._states[str(state)]
            runtime.Board_State = self._state
        else:
            reward = -2
            if runtime.DEBUG:
                print(f"Failed to Mark '{mark}' in Square: {next_move}")
        self._state.print_state()
        return self._state,reward,self._state



class State:
    def __init__(self, term=False, winner=None, board=None):
        self.TERM = term
        self.WINNER = winner
        self.BOARD = board

    def is_over(self):
        return self.TERM

    def is_draw(self):
        return self.TERM and self.WINNER is None

    def get_winner(self):
        return self.WINNER

    def print_state(self):
        state = self.BOARD
        if not isinstance(self.BOARD, list):
            state = eval(self.BOARD)

        a = ' ____ ____ ____'
        state_str = ""
        for i in range(3):
            state_str += f"{a}\n"
            for j in range(3):
                if j == 0:
                    state_str += "|"
                mark = state[i][j]
                if mark is None:
                    mark = ' '
                state_str += f" {mark}  |"
            state_str += f"\n"
        state_str = f"{state_str}{a}"
        print(state_str)
        print(f"TERM: {self.TERM}")
        if self.TERM:
            print(f"WINNER: {self.WINNER}")
        print("\n")

    def mark(self, next_move, mark):
        return Environment.mark(next_move,mark,self.BOARD)
