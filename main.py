import environment
import runtime
import solver
import model_inference


# real_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}

def init_env():
    if runtime.INIT_OBSERVATIONS:
        with open("observations.log", "w") as obs_log:
            obs_log.write("1: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("2: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("3: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("4: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("5: [1,1,1,0,1,1,1,1,0,1]\n")
            obs_log.write("6: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("7: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("8: [1,1,1,1,1,1,1,1,1,1]\n")
            obs_log.write("9: [1,1,1,1,1,1,1,1,1,1]\n")


def main():
    init_env()
    model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    for _ in range(5):
        model_parameters = model_inference.ML(obs_file="observations.log",
                                              prior_model_parameters=model_parameters)
        slvr = solver.Solver(model_parameters, environment, "observations.log")
        slvr.run()


if __name__ == '__main__':
    main()
