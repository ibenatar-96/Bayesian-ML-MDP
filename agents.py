import random


class AiAgent:
    def __init__(self, environment, policy):  # TODO: Add model parameters
        self._mark = 'O'
        self._turn = False
        self._environment = environment
        self._policy = policy

    def play(self, action_parameter=None):
        if self._environment.get_state().is_over():
            return None, None, None
        agent_move = self.choose_action(self._environment.get_state(),
                                        self._environment.get_possible_moves(self._environment.get_state()), self._policy)
        next_state, reward = self._environment.mark(agent_move, 'O')
        return agent_move, next_state, reward

    @staticmethod
    def choose_action(state, possible_moves, policy):
        Q_values = [policy[str(state.BOARD), _action] for _action in possible_moves]
        max_Q = max(Q_values)
        if Q_values.count(max_Q) > 1:
            best_moves = [i for i in range(len(possible_moves)) if Q_values[i] == max_Q]
            i = random.choice(best_moves)
        else:
            i = Q_values.index(max_Q)
        return possible_moves[i]
