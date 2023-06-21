import environment
import utils
import solver
import model_inference
import os


# real_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

def init_env():
    if not os.path.isfile(utils.LOG_FILE):
        print(f"Creating Observations Log File: {utils.LOG_FILE}")
        with open(utils.LOG_FILE, "w+") as lf:
            lf.write("[]")


def main():
    # init_env()
    # for _ in range(5):
    #     model_parameters = model_inference.ML(runtime.LOG_FILE,
    #                                           prior_model_parameters=model_parameters)
    slvr = solver.Solver(utils.REAL_MODEL_PARAMETERS, environment, utils.LOG_FILE)
    slvr.run()


if __name__ == '__main__':
    main()
