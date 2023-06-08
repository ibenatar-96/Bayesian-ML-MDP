import runtime
from random import sample
import solver


class aiAgent:
    def __init__(self, model_parameters=None): # TODO: Add model parameters
        self._mark = 'O'
        self._turn = False
        self.model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
        self._solver = solver.Solver(self.model_parameters)
        self._policy = self._solver.solve()

    def next_move(self, action_parameter=None):
        n_move = self._policy[runtime.Board_State]['Action']
        return n_move
        # if action_parameter is None:
        #     return
        # if action_parameter == 0:
        #     # need to wait for human to mark
        #     pass
        # else:
        #     b_v = dist.Bernoulli(self.model_parameters[action_parameter]).sample()
        #     print(b_v)
