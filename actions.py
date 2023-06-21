import random
import copy

import utils


class Actions:
    def __init__(self, environment=None):
        self._action_utils = [(self.opponent_mark, None), (self.ai_mark, 1), (self.ai_mark, 2), (self.ai_mark, 3),
                              (self.ai_mark, 4),
                              (self.ai_mark, 5), (self.ai_mark, 6),
                              (self.ai_mark, 7), (self.ai_mark, 8), (self.ai_mark, 9)]
        self.environment = environment

    def get_actions(self):
        return map(lambda x: (x[0].__name__, x[1]), self._action_utils)

    def get_possible_actions(self, state):
        """
        :param: State can be a String representation of the current state, or the State Object.
        :param: Action ind is the action index to choose from the action utils field.
        """
        possible_actions = []
        possible_moves = self.environment.get_possible_moves(state)
        for (func_action, action_param) in self.get_actions():
            if func_action in utils.IGNORE_ACTIONS:
                continue
            if action_param in possible_moves:
                possible_actions.append((func_action, action_param))
        return possible_actions

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
        """
        :param: State can be a String representation of the current state, or the State Object.
        :param: Action ind is the action index to choose from the action utils field.
        """
        if not isinstance(state, str):
            state = copy.deepcopy(state.BOARD)
        (func_action, action_parameter) = self._action_utils[action_ind]
        next_state, reward = func_action(state, action_parameter)
        return (func_action, action_parameter), next_state, reward
