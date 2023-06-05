import copy
import time
from functools import reduce
import runtime
from itertools import product
from itertools import combinations


def extract_b_dict(state):
    return {key: value for key, value in state.items() if key.startswith("B")}


class TicTacBoard:
    def __init__(self):
        self._states = self._init_states()
        self._state = self._states[str([[None] * 3 for _ in range(3)])]

    def __iter__(self):
        return iter(self._states)

    def __str__(self):
        output = ""
        for state in self._states:
            output += str(state) + "\n"
        return output

    def _init_states(self):
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
        if ((board[0][0] == board[1][1] == board[2][2]) or (board[0][2] == board[1][1] == board[2][0])) and board[1][
            1] is not None:
            return True, board[1][1]
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    return False, None
        return True, None

    def get_state(self):
        return self._state

    def get_possible_moves(self, mark):
        moves = []
        state = runtime.Board_State.BOARD
        for i in range(3):
            for j in range(3):
                if state[i][j] is None:
                    possible_move = copy.deepcopy(state)
                    possible_move[i][j] = mark
                    moves.append(possible_move)
        return moves

    def mark(self, next_move):
        if not isinstance(next_move, str):
            next_move = str(next_move)
        self._state = self._states[next_move]
        runtime.Board_State = self._state
        if runtime.DEBUG:
            State.print_state(self._state)


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
            state = list(self.BOARD)

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
