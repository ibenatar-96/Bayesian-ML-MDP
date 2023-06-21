import os.path
import utils
import copy
from users import Human
from agents import AiAgent
import actions
import numpyro
import numpyro.distributions as dist
import jax
import time
import random
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from datetime import timedelta
import actions


class Solver:
    """
    Solver, in charge of computing Online planner using Q-Learning
    run() is used for running and testing policy computed.
    """

    def __init__(self, model_parameters, environment, logger):
        self._mapping = {}
        self._model_parameters = model_parameters
        self._environment = environment
        self._logger = logger

    def run(self):
        """
        Running (Playing Tic-Tac-Toe) using the policy computed in solve() function.
        AI Agent - Uses computed Q-Learning policy to choose which cell to mark next.
        Human (Opponent) - Move is drawn (uniformly) random from all possible actions (empty cells).
        Plays runtime.GAMES_TEST games, each game is played until State is Terminal (Winner or Draw).
        """
        policy = self.solve(self._model_parameters)
        board = self._environment.TicTacToe()  # This Board has REAL PARAMETERS!
        self.test_play_games(num_of_games=utils.GAMES_TEST, policy=policy)
        _actions = actions.Actions(board)
        won_games = 0
        run_logger = []
        for _ in tqdm(range(utils.GAMES_COLLECT), desc='Playing Real Tic-Tac-Toe & Collecting Logs..'):
            episode_log = []
            board.reset()
            i = 0
            while not board.get_state().is_over():
                prev_board = copy.deepcopy(board.get_state().BOARD)
                action_ind = self.choose_action(board.get_state(), board.get_possible_moves(), policy, i)
                print(f"action: {action_ind}")
                (func_action, action_parameter), next_state, reward = _actions.activate_action(state=board.get_state(),
                                                                                               action_ind=action_ind)
                episode_log.append((prev_board, (func_action, action_parameter), next_state.BOARD))
                i = (i + 1) % 2
                if board.get_state().is_over() and board.get_state().get_winner() == 'O':
                    won_games += 1
        print(f"Total Games Won: {won_games}/{utils.GAMES_TEST}")
        self.update_observations(run_logger)

    def test_play_games(self, num_of_games, policy):
        print("Agent Playing REAL PARAMETERS Tic-Tac-Toe")
        board = self._environment.TicTacToe()  # This Board has REAL PARAMETERS!
        won_games = 0
        opponent = Human(board)
        ai_agent = AiAgent(board, policy)
        print(f"\n\tTesting {num_of_games} Tic-Tac-Toe Games")
        for _ in tqdm(range(num_of_games), desc='Agent Playing Tic-Tac-Toe..'):
            board.reset()
            while not board.get_state().is_over():
                opponent.play()
                ai_agent.play()
            if board.get_state().get_winner() == 'O':
                won_games += 1
        print(f"\tTotal Games Won: {won_games}/{utils.GAMES_TEST}")
        # return won_games

    def solve(self, model_parameters):
        """
        param: model_parameters: mapping between cell and probability of successfully marking in that cell,
        i.e. {1: 1.0, 2: 1.0, 3: 0.7, 4: 1.0, 5: 0.5, ... , 9: 1.0} Q-Learning algorithm to compute policy.
        Return value - Policy Q, that contains a mapping between (state, action) to expected value in case of taking that
        action from that particular state.

        Solve interacts with 'fabricated' environment, where it assumes the parameters are the model_parameters.
        So basically the Q-Learning is done using the model_parameters transition probabilities.
        """
        board = self._environment.TicTacToe(model_parameters=model_parameters)  # This Board has 'Fictive' PARAMETERS!
        Q = {}
        for state in board.get_states().values():
            for action in board.get_possible_moves(state):
                Q[(str(state.BOARD), action)] = 0.0
        alpha = utils.ALPHA
        epsilon = utils.EPSILON
        discount_factor = utils.DISCOUNT_FACTOR
        iterations = utils.ITERATIONS

        def __choose_action(_state, _available_moves, _mark):
            if random.uniform(0, 1) < epsilon:
                return random.choice(_available_moves)
            else:
                Q_values = [__get_Q_value(_state, _action) for _action in _available_moves]
                if _mark == "O":
                    max_Q = max(Q_values)
                else:
                    max_Q = min(Q_values)
                if Q_values.count(max_Q) > 1:
                    best_moves = [i for i in range(len(_available_moves)) if Q_values[i] == max_Q]
                    i = random.choice(best_moves)
                else:
                    i = Q_values.index(max_Q)
            return _available_moves[i]

        def __get_Q_value(_state, _action):
            if (str(_state.BOARD), _action) not in Q:
                raise Exception(f"State: {_state.BOARD}, Action: {_action} not in Q")
            elif (str(_state.BOARD), _action) in Q and Q[(str(_state.BOARD), _action)] is None:
                Q[(str(_state.BOARD), _action)] = 0.0
            return Q[(str(_state.BOARD), _action)]

        def __update_Q_value(_state, _action, _reward, _next_state, _board):
            next_Q_values = [__get_Q_value(_next_state, next_action) for next_action in
                             _board.get_possible_moves(_next_state)]
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            Q[(str(_state.BOARD), _action)] += alpha * (
                    _reward + discount_factor * max_next_Q - Q[(str(_state.BOARD), _action)])

        start_time = time.time()
        games_won_over_time = []
        marks = ['X', 'O']
        for _ in tqdm(range(iterations), desc='Agent Q-Learning'):
            board.reset()
            i = 0
            while not board.get_state().is_over():
                mark = marks[i % 2]
                second_mark = marks[i % 2]
                state = copy.deepcopy(board.get_state())
                available_moves = board.get_possible_moves(state)
                action = __choose_action(state, available_moves, mark)
                next_state, reward = board.mark(action, mark)
                if not next_state.is_over():
                    state_ = copy.deepcopy(next_state)
                    available_moves_ = board.get_possible_moves(state_)
                    action_ = __choose_action(state_, available_moves_, second_mark)
                    next_state_, reward = board.mark(action_, second_mark)
                __update_Q_value(state, action, reward, next_state, board)
                board.update_state(next_state)
                i += 1
        self.write_to_file(os.path.join("logs", "GAMES_WON_RATIO.txt"), games_won_over_time)
        end_time = time.time()
        print(f"Total Time: {timedelta(seconds=(end_time - start_time))}")
        if utils.PLOT:
            self._plot_win_ratio(games_won_over_time)
        self._save_policy(Q, os.path.join("logs", "Q_VALUES.txt"))
        return Q

    def get_policy(self):
        return self._mapping

    def update_observations(self, run_logger):
        with open(self._logger, "r") as obs_log:
            content = obs_log.read()
            existing_list = eval(content)
            existing_list.append(run_logger)
        with open(self._logger, 'w') as log_file:
            log_file.write(str(existing_list))

    @staticmethod
    def choose_action(state, available_moves, Q, i):
        if i == 0:
            return 9
        Q_values = [Q[str(state.BOARD), action] for action in available_moves]
        max_Q = max(Q_values)
        if Q_values.count(max_Q) > 1:
            best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
            i = random.choice(best_moves)
        else:
            i = Q_values.index(max_Q)
        return available_moves[i]

    @staticmethod
    def load_observations(log_file):
        logger = []
        pass

    @staticmethod
    def _save_policy(Q, txt_file):
        jsonifable = {f"{str(state)}:{str(action)}": val for (state, action), val in Q.items()}
        with open(txt_file, "w") as qd:
            json.dump(jsonifable, qd)

        with open(os.path.join("logs", "Q_DEBUG.txt"), "w") as f:
            list_of_strings = [f'{key[0]}, {key[1]}: {Q[key]}' for key in Q.keys() if Q[key] is not None]
            [f.write(f'{st}\n') for st in list_of_strings]

    @staticmethod
    def _load_policy(txt_file):
        with open(txt_file, "r") as f:
            policy = json.load(f)
        d = {}
        for key, val in policy.items():
            start, end = key.split(':')
            d[str(start), int(end)] = val
        return d

    @staticmethod
    def _plot_win_ratio(games_won_over_time):
        plt.plot(range(0, utils.ITERATIONS, int(utils.ITERATIONS / 10)), games_won_over_time)
        plt.xlabel('Iterations')
        plt.ylabel('Games Won')
        plt.title('Games Won Over Time')
        plt.show()

    @staticmethod
    def write_to_file(file, data):
        with open(file, "w") as f:
            f.write(str(data))
