import runtime
import random


class Human:
    def __init__(self, environment):
        self._mark = 'X'
        self._environment = environment

    def play(self):
        if self._environment.get_state().is_over():
            return
        p_moves = runtime.TicTacToe.get_possible_moves()
        n_move = random.choice(p_moves)
        self._environment.mark(n_move, 'X')
        pass
