import environment
import utils
import solver
import model_inference
import loggers

# real_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}


def main():
    utils.init_model_params()
    logger = loggers.Loggers(utils.LOG_FILE, utils.GAMES_WIN_RATIO_FILE)
    model_parameters = utils.INIT_MODEL_PARAMETERS
    for _ in range(5):
        slvr = solver.Solver(model_parameters, environment, logger)
        slvr.run()
        model_parameters = model_inference.bayesian_learning(utils.LOG_FILE, model_parameters)


if __name__ == '__main__':
    main()
