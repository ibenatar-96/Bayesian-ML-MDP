import environment
import runtime
import solver
import model_inference


def init_env():
    pass


def main():
    init_env()
    prior_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    posterior_model_parameters = model_inference.ML(obs_file="observations.log",
                                                    prior_model_parameters=prior_model_parameters)

    # model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.9, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    # real_model_parameters = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.3, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    # print(f"####### TRAIN & TEST WITH {model_parameters} #######")
    slvr = solver.Solver(posterior_model_parameters, environment, "observations.log")
    slvr.run()
    posterior_model_parameters = model_inference.ML(obs_file="observations.log",
                                                    prior_model_parameters=posterior_model_parameters)
    #
    # print(f"####### TRAIN & TEST WITH {real_model_parameters} #######")
    # slvr = solver.Solver(real_model_parameters, environment)
    # slvr.run()


if __name__ == '__main__':
    main()
