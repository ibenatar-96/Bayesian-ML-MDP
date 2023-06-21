import random


class Actions:
    def __init__(self, environment):
        self._action_utils = [(self.ai_mark, 1), (self.ai_mark, 2), (self.ai_mark, 3), (self.ai_mark, 4),
                              (self.ai_mark, 5), (self.ai_mark, 6),
                              (self.ai_mark, 7), (self.ai_mark, 8), (self.ai_mark, 9), (self.opponent_mark, None)]
        self.environment = environment

    def opponent_mark(self, state, action_parameters):
        if self.environment.get_state().is_over():
            return None, None, None
        p_moves = self.environment.get_possible_moves()
        n_move = random.choice(p_moves)
        return self.environment.mark(n_move, 'X', state)

    def ai_mark(self, state, action_parameter):
        i = (action_parameter - 1) // 3
        j = (action_parameter - 1) % 3
        assert i * 3 + j + 1 == action_parameter
        assert (state[i][j] is None)
        return self.environment.mark(action_parameter, 'O', state)

    def activate_action(self, state, action_ind):
        state.print_state()
        if not isinstance(state, str):
            state = state.BOARD
        (func_action, action_parameter) = self._action_utils[action_ind]
        next_state, reward = func_action(state, action_parameter)
        return (func_action, action_parameter), next_state, reward
