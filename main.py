import Agents
import Users
import Board
from Agents import *
import runtime


def init_env():
    runtime.aiAgent = Agents.aiAgent()
    runtime.Opponent = Users.Human()
    runtime.TicTacBoard = Board.TicTacBoard()
    runtime.Board_State = runtime.TicTacBoard.get_state()


def main():
    state = [['O', 'O', 'X'], ['X', 'X', 'X'], ['O', 'X', 'O']]

    init_env()
    while not runtime.Board_State.is_over():
        runtime.TicTacBoard.mark(runtime.Opponent.next_move())
        if runtime.Board_State.is_over():
            break
        runtime.TicTacBoard.mark(runtime.aiAgent.next_move())

if __name__ == '__main__':
    main()
