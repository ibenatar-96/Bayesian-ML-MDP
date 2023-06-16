import random


class AiAgent:
    def __init__(self, environment, policy, model_parameters=None):  # TODO: Add model parameters
        self._mark = 'O'
        self._turn = False
        self.model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
        self._environment = environment
        self._policy = policy
        # self._policy = self._solver.solve()

    def play(self, action_parameter=None):
        if self._environment.get_state().is_over():
            return
        agent_move = self.choose_action(self._environment.get_state(),
                                        self._environment.get_possible_moves(self._environment.get_state()), self._policy)
        self._environment.mark(agent_move, 'O')

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
