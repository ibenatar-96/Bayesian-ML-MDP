import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import runtime
import time


class Actions:
    def __init__(self):
        self._action_utils = [(self.ai_mark, {1}), (self.ai_mark, {2}), (self.ai_mark, {3}), (self.ai_mark, {4}),
                              (self.ai_mark, {5}), (self.ai_mark, {6}),
                              (self.ai_mark, {7}), (self.ai_mark, {8}), (self.ai_mark, {9}), (self.human, {})]

    def human(self, state, action_parameters, model_parameters):
        p_moves = runtime.TicTacToe.get_possible_moves()
        distribution = dist.Categorical(jnp.ones(len(p_moves)) / len(p_moves))
        y = numpyro.sample('y', distribution).item()

    def ai_mark(self, state, action_parameter, model_parameters):
        i = (action_parameter - 1) // 3
        j = (action_parameter - 1) % 3
        assert i * 3 + j + 1 == action_parameter
        assert (state[i][j] is None)
        # y = numpyro.sample('y', dist.Bernoulli(probs=model_parameters[action_parameter]),
        #                    rng_key=jax.random.PRNGKey(int(time.time() * 1E6))).item()
        # if y > 0:
        return runtime.TicTacToe.mark(action_parameter, 'X', state)

    def activate_action(self, state, action_ind, model_parameters):
        func_action, action_parameter = self._action_utils[action_ind]
        next_state, reward, observation = func_action(state, action_parameter, model_parameters)
