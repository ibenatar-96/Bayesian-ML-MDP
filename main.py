import agents
import users
import environment
import runtime
import solver


def init_env():
    real_model_parameters = {1: 0.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.2, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    real_model_parameters_2 = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.6, 5: 0.2, 6: 0.7, 7: 0.1, 8: 0.5, 9: 0.6}

    runtime.Environment = environment.Environment(real_model_parameters_2)
    runtime.Board_State = runtime.Environment.get_state()
    runtime.aiAgent = agents.AiAgent()
    runtime.Opponent = users.Human()


def main():
    init_env()
    model_parameters = {1: 0.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
    slvr = solver.Solver(model_parameters)
    slvr.run()
    # while not runtime.Board_State.is_over():
    #     runtime.Environment.mark(runtime.Opponent.next_move(), 'X')
    #     if runtime.Board_State.is_over():
    #         break
    #     runtime.Environment.mark(runtime.aiAgent.next_move(), 'O')


if __name__ == '__main__':
    main()
