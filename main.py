import environment
import utils
import solver
import model_inference
import os
import actions


# real_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

def init_env():
    """
    Starts the env - Creates Observation File if it does not exist,
    Also initializes the utils.INIT_MODEL_PARAMETERS, which is the initial belief of the probability for each action.
    an action is composed of a tuple (function name, action parameter),
    For example: ('ai_mark', 2) means,
    The action 'ai_mark' - it is the AI models' turn to mark, and it marks the 2nd cell.
    utils.INIT_MODEL_PARAMETERS = {
    ('ai_mark', 1): 1.0
    ('ai_mark', 2): 1.0
    ...
    ('ai_mark', 5): 0.8
    ...}
    """
    if not os.path.isfile(utils.LOG_FILE) or utils.CLEAN_OBS:
        print(f"Creating / Cleaning Observations Log File: {utils.LOG_FILE}")
        open(utils.LOG_FILE, 'w').close()

    for (func_action, action_param) in actions.Actions().get_actions():
        if func_action not in utils.IGNORE_ACTIONS and (func_action, action_param) not in utils.INIT_MODEL_PARAMETERS:
            utils.INIT_MODEL_PARAMETERS[(func_action, action_param)] = 1.0


def main():
    init_env()
    model_parameters = utils.INIT_MODEL_PARAMETERS
    for _ in range(5):
        slvr = solver.Solver(model_parameters, environment, utils.LOG_FILE)
        slvr.run()
        model_parameters = model_inference.ML(utils.LOG_FILE, model_parameters)


if __name__ == '__main__':
    main()
