import runtime
import environment
import copy


class Solver:
    """
    This is the solver...
    """
    def __init__(self, model_parameters):
        self._mapping = {}
        self._model_parameters = model_parameters

    def solve(self): # This is solver
        mapping = {state: {'Board': state.BOARD, 'Value': float('-inf'), 'Action': None} for state in
                   runtime.TicTacBoard.get_states().values()}

        for _ in range(runtime.ITERATIONS):
            delta = 0
            for state in mapping.keys():
                if state.is_over():
                    if state.get_winner() == 'O':
                        reward = runtime.WIN_REWARD
                    elif state.get_winner() == 'X':
                        reward = runtime.LOSE_REWARD
                    else:
                        reward = runtime.DRAW_REWARD
                    mapping[state].update({'Value': reward, 'Action': None})
                    continue

                new_value = runtime.MIN
                best_action = None

                for action_o in runtime.TicTacBoard.get_possible_moves(state):
                    next_board = copy.deepcopy(state.BOARD)
                    i = (action_o - 1) // 3
                    j = (action_o - 1) % 3
                    next_board[i][j] = 'O'
                    state_o = runtime.TicTacBoard.get_states()[str(next_board)]

                    if state_o.is_over():
                        if state_o.get_winner() == 'O':
                            expected_return = runtime.WIN_REWARD
                        elif state_o.get_winner() == 'X':
                            expected_return = runtime.LOSE_REWARD
                        else:
                            expected_return = runtime.DRAW_REWARD
                    else:
                        expected_return_x = runtime.MAX  # Initialize with worst-case value for 'X'
                        for action_x in runtime.TicTacBoard.get_possible_moves(state_o):
                            next_next_board = copy.deepcopy(state_o.BOARD)
                            i = (action_x - 1) // 3
                            j = (action_x - 1) % 3
                            next_next_board[i][j] = 'X'
                            state_o_x = runtime.TicTacBoard.get_states()[str(next_next_board)]
                            state_o_x_mapping = mapping[state_o_x]
                            expected_return_x = min(expected_return_x, state_o_x_mapping['Value'])

                        expected_return = expected_return_x
                    expected_return += runtime.IMMEDIATE_REWARD

                    if expected_return > new_value:
                        best_action = action_o
                        new_value = expected_return

                        if state_o.is_over() and state_o.get_winner() == 'O':
                            break

                delta = max(delta, abs(mapping[state]['Value'] - new_value))
                mapping[state].update({'Value': new_value, 'Action': best_action})

            if delta < runtime.THETA:
                break

        return mapping

    def get_policy(self):
        return self._mapping
