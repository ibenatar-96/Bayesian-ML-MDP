import Agents
import Users
import Board
from Agents import *
import runtime


def init_env():
    runtime.Agent = Agents.Agent()
    runtime.Opponent = Users.Human()
    runtime.Board = Board.Board()


def world_state():
    pass


def all_terminated():
    pass


def terminate_agents():
    pass


def main():
    init_env()


if __name__ == '__main__':
    main()
