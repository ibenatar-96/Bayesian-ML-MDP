import environment
import runtime
import solver
import model_inference
import os


# real_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

def init_env():
    log_path = os.path.join("logs", "observations.log")
    if not os.path.isfile(log_path):
        with open(log_path, "w+") as lf:
            lf.write("[]")


def main():
    init_env()
    model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    for _ in range(5):
        model_parameters = model_inference.ML(obs_file=os.path.join("logs", "observations.log"),
                                              prior_model_parameters=model_parameters)
        slvr = solver.Solver(model_parameters, environment, os.path.join("logs", "observations.log"))
        slvr.run()


if __name__ == '__main__':
    main()
