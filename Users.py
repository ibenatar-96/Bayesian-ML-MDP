import copy

import runtime
import Minimax


class Human:
    def __init__(self, pos=None, _id=None, limit=10):
        self._pos = pos
        self._next_v = None
        self._path = []
        self._ticks_left = 0
        self._full_path_ticks = 0
        self._terminated = False
        self._limit = limit
        self._found_goal = False
        self._ticks = 0
        self._num_rescued = 0
        self._total_score = 0
        self._expands = 0
        self._expand_time = 0
        self._id = _id
