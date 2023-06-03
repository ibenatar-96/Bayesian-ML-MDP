import copy
from functools import reduce
import runtime
from itertools import product


def extract_b_dict(state):
    return {key: value for key, value in state.items() if key.startswith("B")}


class TicTacBoard:
    def __init__(self):
        self._states = self._init_states()
        self._state = [[None] * 3 for _ in range(3)]
        if runtime.DEBUG:
            print(f"Init State:\n"
                  f"{self.state_string(self._state)}")

    def __iter__(self):
        return iter(self._states)

    def __str__(self):
        output = ""
        for state in self._states:
            output += str(state) + "\n"
        return output

    def _init_states(self):
        states = {}
        square_mark = ['X', 'O', None]
        for row1 in product(square_mark, repeat=3):
            state = [list(row1), [None] * 3, [None] * 3]
            term, win = self._is_terminal(state)
            if term:
                states[str(state)] = [True, win]
                continue
            for row2 in product(square_mark, repeat=3):
                state = [list(row1), list(row2), [None] * 3]
                term, win = self._is_terminal(state)
                if term:
                    states[str(state)] = [True, win]
                    continue
                for row3 in product(square_mark, repeat=3):
                    state = [list(row1), list(row2), list(row3)]
                    term, win = self._is_terminal(state)
                    if term:
                        states[str(state)] = [True, win]
                    else:
                        states[str(state)] = [False, None]
        return states

    def _is_terminal(self, board):
        for row in board:
            player = row[0]
            if all(i == player for i in row) and player is not None:
                return True, player
        for j in range(3):
            column = [row[j] for row in board]
            player = column[0]
            if all(i == player for i in column) and player is not None:
                return True, player
        if (board[0][0] == board[1][1] == board[2][2] or board[0][2] == board[1][1] == board[2][0]) and board[1][
            1] is not None:
            return True, board[0][0]
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    return False, None
        return True, None

    def state_string(self, state):
        if not isinstance(state, list):
            state = list(state)

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
                state_str += f"{mark}   |"
            state_str += f"\n"
        state_str = f"{state_str}{a}"
        return state_str

    def get_state(self):
        return self._state

    def get_possible_moves(self, mark):
        moves = []
        state = runtime.World_State
        for i in range(3):
            for j in range(3):
                if state[i][j] is None:
                    possible_move = copy.deepcopy(state)
                    possible_move[i][j] = mark
                    moves.append(possible_move)
        return moves


class State:
    def __init__(self, state=None, win=False, draw=False):
        self.WIN = win
        self.DRAW = draw
