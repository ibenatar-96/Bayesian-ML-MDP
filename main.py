import Agents
import Users
import Board
from Agents import *
import runtime


def init_env():
    runtime.TicTacBoard = Board.TicTacBoard()
    runtime.Board_State = runtime.TicTacBoard.get_state()
    runtime.aiAgent = Agents.aiAgent()
    runtime.Opponent = Users.Human()


def main():
    init_env()
    while not runtime.Board_State.is_over():
        runtime.TicTacBoard.mark(runtime.Opponent.next_move(), 'X')
        if runtime.Board_State.is_over():
            break
        runtime.TicTacBoard.mark(runtime.aiAgent.next_move(), 'O')


if __name__ == '__main__':
    main()
