import agents
import users
import environment
from agents import *
import runtime


def init_env():
    runtime.TicTacBoard = environment.Environment()
    runtime.Board_State = runtime.TicTacBoard.get_state()
    runtime.aiAgent = agents.aiAgent()
    runtime.Opponent = users.Human()


def main():
    init_env()
    while not runtime.Board_State.is_over():
        runtime.TicTacBoard.mark(runtime.Opponent.next_move(), 'X')
        if runtime.Board_State.is_over():
            break
        runtime.TicTacBoard.mark(runtime.aiAgent.next_move(), 'O')


if __name__ == '__main__':
    main()
