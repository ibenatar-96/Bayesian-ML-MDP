import copy
import sys
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


class TicTacToe:
    """
    Tic Tac Toe board with probabilities to mark in each square
    3x3 Matrix, each cell: 'X', 'O', None
    2 Players - AI Agent, and Human
    AI Agent - 'O'; chooses next square depending on policy from Q-Learning
    Human - 'X'; chooses next square with Uniform Prob.
    Terminal State = Winner (3 in a row / column / diagonal) or Full Board (no empty square) and Draw
    """

    def __init__(self, model_parameters=runtime.REAL_MODEL_PARAMETERS):
        self._model_parameters = model_parameters
        self._states = self._init_states_space()
        self._state = self._states[str([[None] * 3 for _ in range(3)])]
        runtime.ORIGINAL_STATES = copy.deepcopy(self._states)

    def __iter__(self):
        return iter(self._states)

    def __str__(self):
        output = ""
        for state in self._states:
            output += str(state) + "\n"
        return output

    def _init_states_space(self):
        """
        Creates State Space for all possible states, 3^9 states possible (3 options for each of 9 cells - 'X', 'O' or None).
        States is a mapping between state is which is presented by String of the board (i.e. - [['X','O',None],
                                                                                               ['O',None,None],
                                                                                               ['X','O','X']])
        to State Objects which are defined by - TERM: Whether or not state is terminal state (exists winner / full board and draw)
                                                WINNER: Who is the winner of this State Object - 'X' / 'O' / None
                                                BOARD: String representation of the board (same as example above).
        """
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
        """
        Checks whether board is terminal - (3 in a row / column / diagonal) or Full Board (no empty square) and Draw.
        Return value: 2 values - Terminal (Boolean) and Winner ('X' / 'O' / None)
        """
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

    def update_state(self, state):
        self._state = self.get_states()[str(state.BOARD)]
        # self.get_state().print_state()

    def get_possible_moves(self, state=None):
        """
        Returns a list of all possible moves - possible moves are all cells marked as None.
        """
        _state = state
        if _state is None:
            _state = self.get_state()
        if isinstance(_state, State):
            _state = _state.BOARD
        moves = [(i * 3 + j + 1) for i in range(3) for j in range(3) if _state[i][j] is None]
        return moves

    def get_real_model_parameters(self):
        return self._model_parameters

    def mark(self, next_move, mark, state=None):
        """
        Marks cell {next_move} with probability self._model_parameters[{next_move}].
        i.e. self._model_parameters = {1: 1.0, 2: 1.0, 3: 0.7, 4: 1.0, 5: 0.5, ... , 9: 1.0} and next_move = 3, mark = 'O',
        so with probability 0.7 cell 3 will be marked with 'O'.
        Return Value: returns Reward depending on ending state (State can change if sample value is < prob, else stays the same).
                      Also returns the Current State (New State if changed, last State if didn't change).
        """
        reward = -1
        if state is None:
            state = copy.deepcopy(self.get_state().BOARD)
        i = (next_move - 1) // 3
        j = (next_move - 1) % 3
        assert i * 3 + j + 1 == next_move
        assert (state[i][j] is None)
        y = numpyro.sample('y', dist.Bernoulli(probs=self._model_parameters[next_move]),
                           rng_key=jax.random.PRNGKey(int(time.time() * 1E6))).item()
        if y > 0 or mark == 'X':
            state[i][j] = mark
            new_state = self.get_states()[str(state)]
            if new_state.is_over() and new_state.get_winner() == 'X':
                reward += runtime.LOSE_REWARD
            elif new_state.is_over() and new_state.get_winner() == 'O':
                reward += runtime.WIN_REWARD
            elif new_state.is_over():
                reward += runtime.DRAW_REWARD
            self.update_state(new_state)
            # runtime.Board_State = new_state
        else:
            reward = -2
            if runtime.DEBUG:
                print(f"Failed to Mark '{mark}' in Square: {next_move}")
        if runtime.DEBUG_BOARD:
            self.get_state().print_state()
        return self.get_state(), reward

    def reset(self):
        init_str_state = str([[None] * 3 for _ in range(3)])
        init_state = self._states[init_str_state]
        self.update_state(init_state)
        if runtime.DEBUG_BOARD:
            print("Reset:")
            self._state.print_state()


class State:
    """
    State Object, is composed of: TERM - (Boolean) Is the State Object a TERMINAL State.
                                  WINNER - 'X', 'O', None. Depending on the board.
                                  BOARD - String representation of the state board, i.e.       [['X','O',None],
                                                                                               ['O',None,None],
                                                                                               ['X','O','X']])
    TERMINAL STATE:  True = 3 in a row / diagonal / column, or No empty cell & No winner. False = Every other case.
    """
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

    def print_state(self, file=sys.stdout):
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
        print(state_str, file=file)
        print(f"TERM: {self.TERM}", file=file)
        if self.TERM:
            print(f"WINNER: {self.WINNER}", file=file)
        print("\n", file=file)
