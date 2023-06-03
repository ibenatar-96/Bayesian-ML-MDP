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
    init_env()
    runtime.Opponent.move()
    runtime.aiAgent.move(1)

if __name__ == '__main__':
    main()
