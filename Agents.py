import runtime
from random import sample

class aiAgent:
    def __init__(self):
        self._mark = 'O'
        self._turn = False
        self.model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

    def next_move(self, action_parameter=None):
        p_moves = runtime.TicTacBoard.get_possible_moves(self._mark)
        n_move = sample(p_moves, 1)
        return n_move[0]
        # if action_parameter is None:
        #     return
        # if action_parameter == 0:
        #     # need to wait for human to mark
        #     pass
        # else:
        #     b_v = dist.Bernoulli(self.model_parameters[action_parameter]).sample()
        #     print(b_v)
