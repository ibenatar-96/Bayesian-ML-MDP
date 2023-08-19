import os.path
import utils
import copy
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
        # Step 1 - Computes Optimal Policy given model parameters, and optimal adversary
        policy = self.compute_policy(self._model_parameters)

        # Step 2 - Evaluates computed policy on real environment and real (random) adversary
        self.evaluate_policy(num_of_games=utils.GAMES_TEST, policy=policy)

        # Step 3 - Executes computed policy on real adversary, and collects statistics
        board = self._environment.TicTacToe()  # This Board has REAL PARAMETERS!
        actions_ = actions.Actions(board)
        won_games = 0
        run_logger = []
        print(f"\n\tPlaying Real & Collecting Logs {utils.GAMES_COLLECT} Tic-Tac-Toe Games")
        for _ in tqdm(range(utils.GAMES_COLLECT), desc='Playing Real Tic-Tac-Toe & Collecting Logs..'):
            episode_log = []
            board.reset()
            i = 0
            while not board.get_state().is_over():
                prev_board = copy.deepcopy(board.get_state().BOARD)
                action_ind = self.choose_action(board.get_state(), actions_.get_possible_actions(prev_board), policy, i)
                (func_action, action_parameter), next_state, reward = actions_.activate_action(state=board.get_state(),
                                                                                               action_ind=action_ind)
                episode_log.append((prev_board, (func_action.__name__, action_parameter), next_state.BOARD))
                i = (i + 1) % 2
                if board.get_state().is_over() and board.get_state().get_winner() == 'O':
                    won_games += 1
            run_logger.append(episode_log)
        print(f"\tTotal Games Won: {won_games}/{utils.GAMES_COLLECT}")
        self.update_observations(run_logger)

    def evaluate_policy(self, num_of_games, policy, txt_file=None):
        """
        Evaluates Accuracy of policy on real parameters env.
        Plays {num_of_games} Tic-Tac-Toe games, with REAL MODEL parameters, meaning that it tests the accuracy of our AI
        Model, given the Policy.
        Tests how our trained AI Model (using our belief model parameters) performs in the REAL PARAMETER env.
        For example, our AI Model can be trained on an env. where the probability to mark at cell 5 is 0.8, and a policy
        Q can be computed with this probability, then we test this policy Q on a real env, where the real probability
        for marking cell 5 is 0.3.
        """
        print("Agent Playing REAL PARAMETERS Tic-Tac-Toe")
        board = self._environment.TicTacToe()  # This Board has REAL PARAMETERS!
        won_games = 0
        if txt_file is not None:
            open(txt_file, 'w').close()
        actions_ = actions.Actions(board)
        print(f"\n\tTesting {num_of_games} Tic-Tac-Toe Games")
        for _ in tqdm(range(num_of_games), desc='Agent Playing Tic-Tac-Toe..'):
            game_log = []
            board.reset()
            i = 0
            while not board.get_state().is_over():
                prev_board = copy.deepcopy(board.get_state().BOARD)
                action_ind = self.choose_action(board.get_state(), actions_.get_possible_actions(prev_board), policy, i)
                (func_action, action_parameter), next_state, reward = actions_.activate_action(state=board.get_state(),
                                                                                               action_ind=action_ind)
                game_log.append((action_parameter, board.get_state()))
                i = (i + 1) % 2
                if board.get_state().is_over() and board.get_state().get_winner() == 'O':
                    won_games += 1
                elif board.get_state().is_over() and board.get_state().get_winner() != 'O' and txt_file is not None:
                    with open(txt_file, "a") as f:
                        for (action, state) in game_log:
                            print(f"Action: {action}", file=f)
                            state.print_state(file=f)
                        print("------------- NEW GAME ------------\n", file=f)
        print(f"\tTotal Games Won: {won_games}/{num_of_games}")
        return won_games

    def compute_policy(self, model_parameters):
        """
        param: model_parameters: mapping between cell and probability of successfully marking in that cell,
        i.e. {1: 1.0, 2: 1.0, 3: 0.7, 4: 1.0, 5: 0.5, ... , 9: 1.0} Q-Learning algorithm to compute policy.
        Return value - Policy Q, that contains a mapping between (state, action) to expected value in case of taking that
        action from that particular state.

        Solve interacts with 'fabricated' environment, where it assumes the parameters are the model_parameters.
        So basically the Q-Learning is done using the model_parameters transition probabilities.
        """
        board = self._environment.TicTacToe(model_parameters=model_parameters)  # This Board has 'Fictive' PARAMETERS!
        actions_ = actions.Actions(board)
        Q = {}
        for state in board.get_states().values():
            for (func_action, action_param) in actions_.get_possible_actions(state):
                Q[(str(state.BOARD), (func_action, action_param))] = 0.0
        alpha = utils.ALPHA
        epsilon = utils.EPSILON
        discount_factor = utils.DISCOUNT_FACTOR
        iterations = utils.ITERATIONS

        def __choose_action(_state, _available_actions, _mark):
            if random.uniform(0, 1) < epsilon:
                return random.choice(_available_actions)
            else:
                Q_values = [__get_Q_value(_state, _action) for _action in _available_actions]
                if _mark == "O":
                    max_Q = max(Q_values)
                else:
                    max_Q = min(Q_values)
                if Q_values.count(max_Q) > 1:
                    best_moves = [i for i in range(len(_available_actions)) if Q_values[i] == max_Q]
                    i = random.choice(best_moves)
                else:
                    i = Q_values.index(max_Q)
            return _available_actions[i]

        def __get_Q_value(_state, _action):
            if (str(_state.BOARD), _action) not in Q:
                raise Exception(f"State: {_state.BOARD}, Action: {_action} not in Q")
            elif (str(_state.BOARD), _action) in Q and Q[(str(_state.BOARD), _action)] is None:
                Q[(str(_state.BOARD), _action)] = 0.0
            return Q[(str(_state.BOARD), _action)]

        def __update_Q_value(_state, _action, _reward, _next_state, _actions):
            next_Q_values = [__get_Q_value(_next_state, next_action) for next_action in
                             _actions.get_possible_actions(_next_state)]
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            Q[(str(_state.BOARD), _action)] += alpha * (
                    _reward + discount_factor * max_next_Q - Q[(str(_state.BOARD), _action)])

        start_time = time.time()
        games_won = []
        marks = ['X', 'O']
        for _ in tqdm(range(iterations), desc='Agent Q-Learning'):
            board.reset()
            i = 0
            while not board.get_state().is_over():
                mark = marks[i % 2]
                second_mark = marks[(i + 1) % 2]
                state = copy.deepcopy(board.get_state())
                available_actions = actions_.get_possible_actions(state)
                (func_action, action_param) = __choose_action(state, available_actions, mark)
                next_state, reward = board.mark(action_param, mark)
                if not next_state.is_over():
                    state_ = copy.deepcopy(next_state)
                    available_actions_ = actions_.get_possible_actions(state_)
                    (func_action_, action_param_) = __choose_action(state_, available_actions_, second_mark)
                    next_state_, reward = board.mark(action_param_, second_mark)
                __update_Q_value(state, (func_action, action_param), reward, next_state, actions_)
                board.update_state(next_state)
                i += 1
        self.write_to_file("games_won_over_time.log", games_won)
        end_time = time.time()
        print(f"Total Time: {timedelta(seconds=(end_time - start_time))}")
        # if utils.PLOT:
        #     self._plot_win_ratio(games_won_over_time)
        if utils.DEBUG:
            self._save_policy(Q, os.path.join("logs", "Q_VALUES.txt"))
        return Q

    def get_policy(self):
        return self._mapping

    def update_observations(self, run_logger):
        """
        Updates the observation log file, Each line the the observation log file is an episode.
        An episode is a full Tic-Tac-Toc game.
        It is composed of a list of tuples - each tuple is (state, action, next_state).
        and each tuple in the list is an action / play by the AI or Opponent.
        For the opponent plays the tuple will be - (state, 'opponent_mark', next_state).
        For the AI Model plays the tuple will be - (state, {action parameter (1,2,3..,9)}, next_state)
        """
        with open(self._logger, 'a') as log_file:
            for episode in run_logger:
                log_file.write(f"{str(episode)}\n")

    @staticmethod
    def choose_action(state, available_moves, Q, i):
        if i == 0:
            return 0
        Q_values = [Q[str(state.BOARD), action] for action in available_moves]
        max_Q = max(Q_values)
        if Q_values.count(max_Q) > 1:
            best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
            i = random.choice(best_moves)
        else:
            i = Q_values.index(max_Q)
        return available_moves[i][1]

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
            d[str(start), str(end)] = val
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
        with open(file, "a") as f:
            f.write(str(data))
