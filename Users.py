import runtime
from random import sample


class Human:
    def __init__(self):
        self._mark = 'X'

    def next_move(self):
        p_moves = runtime.TicTacBoard.get_possible_moves(self._mark)
        n_move = sample(p_moves, 1)
        return n_move[0]
