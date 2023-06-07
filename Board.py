import copy
import time
from functools import reduce
import runtime
from itertools import product
from itertools import combinations
import ast


class TicTacBoard:
    def __init__(self):
        self._states = self._init_states()
        if runtime.CLEAN_STATES:
            self._clean_states()
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

    def _clean_states(self):
        r = dict(self._states)
        for board in r.keys():
            board = eval(board)
            player_x = player_o = False
            for row in board:
                player = row[0]
                if all(i == player for i in row) and player is not None:
                    if player == 'X':
                        player_x = True
                    elif player == 'O':
                        player_o = True
            for j in range(3):
                column = [row[j] for row in board]
                player = column[0]
                if all(i == player for i in column) and player is not None:
                    if player == 'X':
                        player_x = True
                    elif player == 'O':
                        player_o = True
            if board[0][0] == board[1][1] == board[2][2] and board[1][1] is not None:
                if board[1][1] == 'X':
                    player_x = True
                elif board[1][1] == 'O':
                    player_o = True
            if board[0][2] == board[1][1] == board[2][0] and board[1][1] is not None:
                if board[1][1] == 'X':
                    player_x = True
                elif board[1][1] == 'O':
                    player_o = True

            if player_x and player_o:
                del self._states[str(board)]

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

    def get_states(self):
        return self._states

    @staticmethod
    def get_possible_moves(state=None):
        moves = []
        _state = state
        if _state is None:
            _state = runtime.Board_State.BOARD
        if isinstance(_state, State):
            _state = _state.BOARD
        if not isinstance(_state, list):
            _state = eval(_state)

        moves = [(i * 3 + j + 1) for i in range(3) for j in range(3) if _state[i][j] is None]
        return moves

    def mark(self, next_move, mark):
        state = self._state.BOARD
        if not isinstance(state, list):
            state = eval(state)
        i = (next_move - 1) // 3
        j = (next_move - 1) % 3
        assert i * 3 + j + 1 == next_move
        assert(state[i][j] is None)
        state[i][j] = mark
        self._state = self._states[str(state)]
        runtime.Board_State = self._state
        self._state.print_state()


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
