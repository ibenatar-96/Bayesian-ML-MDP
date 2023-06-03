import numpyro
import numpyro.distributions as dist


class aiAgent:
    def __init__(self):
        self._mark = 'O'
        self._turn = False
        self.model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

    def move(self, action_parameter):
        if action_parameter is None:
            return
        if action_parameter == 0:
            # need to wait for human to mark
            pass
        else:
            b_v = dist.Bernoulli(self.model_parameters[action_parameter]).sample()
            # test
            print(b_v)
