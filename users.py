import runtime
import random


class Human:
    def __init__(self, environment):
        self._mark = 'X'
        self._environment = environment

    def play(self):
        if self._environment.get_state().is_over():
            return None, None, None
        p_moves = self._environment.get_possible_moves()
        n_move = random.choice(p_moves)
        next_state, reward = self._environment.mark(n_move, 'X')
        return n_move, next_state, reward
