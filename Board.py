import copy

import runtime
from itertools import product


def extract_b_dict(state):
    return {key: value for key, value in state.items() if key.startswith("B")}


class Board:
    def __init__(self):
        self._states = self._init_states()
        print("yo")
        # self._init_actions()

    def __iter__(self):
        return iter(self._vector)

    def __str__(self):
        output = ""
        for state in self._vector:
            output += str(state) + "\n"
        return output

    def _init_states(self):
        states = []

        def generate_states(board):
            term, winner = self._is_terminal(board)
            if term:
                return winner, board
            for i in range(3):
                for j in range(3):
                    if board[i][j] is None:
                        new_board_x = copy.deepcopy(board)
                        new_board_o = copy.deepcopy(board)
                        new_board_n = copy.deepcopy(board)
                        new_board_x[i][j] = 'X'
                        new_board_o[i][j] = 'O'
                        new_board_n[i][j] = None
                        states.append(generate_states(new_board_x))
                        states.append(generate_states(new_board_o))
                        states.append(generate_states(new_board_n))

        empty_board = [[None] * 3 for _ in range(3)]
        generate_states(empty_board)
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
        if (board[0][0] == board[1][1] == board[2][2] or board[0][2] == board[1][1] == board[2][0]) and board[1][1] is not None:
            return True,board[0][0]
        for i in range(3):
            for j in range(3):
                if board[i][j] is None:
                    return False, None
        return True, None
