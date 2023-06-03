import copy
import heapq
from Agents import *
import runtime

EXPANDS = 0
LEAFS = 0


class GameTree:
    pass
    #
    # def __init__(self, heuristic_func, score_tuple, _id, limit=None):
    #     state = runtime.World_State
    #     self._root = GameNode(state['agents_positions'], state['populated_v'], state['broken_v'],
    #                           heuristic_func(score_tuple, _id), _id, score_tuple)
    #     self._heuristic_func = heuristic_func
    #     self._limit = limit
    #     if runtime.DEBUG:
    #         print(f"Game Tree Root:\n{self._root.__str__()}")
    #
    # def find_goal(self, agent_id):
    #     global LEAFS
    #     LEAFS = 0
    #     max_path_node = GameTree.minimax(self._root, 0, True, runtime.MIN, runtime.MAX, self._heuristic_func,
    #                                      self._limit, agent_id)
    #     print(f"LEAFS: {LEAFS}")
    #     fixed_path = self.fix_path(max_path_node)
    #     return fixed_path[0].agents_pos[agent_id] if len(fixed_path) > 0 else None
    #
    # @staticmethod
    # def fix_path(node):
    #     fixed_path = []
    #     while node.parent is not None:
    #         fixed_path.insert(0, node)
    #         node = node.parent
    #     if runtime.DEBUG and len(fixed_path) > 0:
    #         print(f"Fixed Path:\n{fixed_path[len(fixed_path)-1]}")
    #     return fixed_path
    #
    # @staticmethod
    # def minimax(node, depth, maximizing_player,
    #             alpha, beta, heuristic_func, limit, agent_id):
    #
    #     if runtime.Active_Game_Type != "Adversarial":
    #         maximizing_player = True
    #
    #     global EXPANDS
    #     EXPANDS += 1
    #     agent_pos = node.agents_pos[node.id]
    #     if depth == limit - 1:
    #         global LEAFS
    #         LEAFS += 1
    #         # print(node.__str__())
    #         return node
    #
    #     if maximizing_player:
    #
    #         best = GameNode(None, None, None, runtime.MIN, None, None)
    #         temp = best
    #         # Recur for left and right children
    #         for neighbor in runtime.World_Graph.get_gdict()[agent_pos].get_adjacent():
    #             _agents_pos = copy.deepcopy(node.agents_pos)
    #             _agents_pos[node.id] = neighbor['id']
    #             _id = (node.id - 1) % 2
    #             _populated_v = node.populated_v if not runtime.World_Graph.get_gdict()[
    #                 neighbor['id']].get_populated() else \
    #                 [v for v in node.populated_v if v != neighbor['id']]
    #             _broken_v = node.broken_v if not runtime.World_Graph.get_gdict()[neighbor['id']].get_brittle() \
    #                 else node.broken_v + [neighbor['id']]
    #             _parent = node
    #             _children = []
    #             # _score1 = node.score_tuple[0]
    #             _score_tuple = copy.deepcopy(node.score_tuple)
    #             if _populated_v != node.populated_v:
    #                 if node.id == 0:
    #                     _score_tuple = (
    #                         _score_tuple[0] + runtime.World_Graph.get_gdict()[neighbor['id']].get_population(),
    #                         _score_tuple[1])
    #                 else:
    #                     _score_tuple = (
    #                         _score_tuple[0],
    #                         _score_tuple[1] + runtime.World_Graph.get_gdict()[neighbor['id']].get_population())
    #             if neighbor['id'] in node.broken_v:
    #                 continue
    #             #     _h = runtime.MIN
    #             # else:
    #             _h = heuristic_func(_score_tuple, agent_id)
    #             neighbor_node = GameNode(_agents_pos, _populated_v, _broken_v, _h, _id, _score_tuple, _parent)
    #             node.children.append(neighbor_node)
    #             ret_n = GameTree.minimax(neighbor_node, depth + 1,
    #                                      False, alpha, beta, heuristic_func, limit, agent_id)
    #             # If Semi-Cooperative and there's a tie
    #             if runtime.Active_Game_Type == "Semi-Cooperative" and ret_n.h == best.h:
    #                 if not best.score_tuple:
    #                     best = ret_n
    #                 else:
    #                     best = ret_n if sum(ret_n.score_tuple) >= sum(best.score_tuple) else best
    #             else:
    #                 if runtime.Active_Game_Type == "Fully-Cooperative":
    #                     best = ret_n if ret_n.h >= best.h else best
    #                 else:
    #                     best = ret_n if ret_n.h > best.h else best
    #             alpha = max(alpha, best.h)
    #
    #             # Alpha Beta Pruning
    #             if runtime.Pruning and beta <= alpha:
    #                 # print(f"pruned node: {neighbor_node}")
    #                 break
    #
    #         # It may occur that all the neighbors are broken
    #         if best == temp:
    #             return node
    #
    #         return best
    #
    #     else:
    #         best = GameNode(None, None, None, runtime.MAX, None, None)
    #         temp = best
    #         # Recur for left and
    #         # right children
    #         for neighbor in runtime.World_Graph.get_gdict()[agent_pos].get_adjacent():
    #             _agents_pos = copy.deepcopy(node.agents_pos)
    #             _agents_pos[node.id] = neighbor['id']
    #             _id = (node.id - 1) % 2
    #             _populated_v = node.populated_v if not runtime.World_Graph.get_gdict()[
    #                 neighbor['id']].get_populated() else \
    #                 [v for v in node.populated_v if v != neighbor['id']]
    #             _broken_v = node.broken_v if not runtime.World_Graph.get_gdict()[neighbor['id']].get_brittle() \
    #                 else node.broken_v + [neighbor['id']]
    #             _parent = node
    #             _children = []
    #             # _score1 = node.score_tuple[0]
    #             _score_tuple = copy.deepcopy(node.score_tuple)
    #             if _populated_v != node.populated_v:
    #                 if node.id == 0:
    #                     _score_tuple = (
    #                         _score_tuple[0] + runtime.World_Graph.get_gdict()[neighbor['id']].get_population(),
    #                         _score_tuple[1])
    #                 else:
    #                     _score_tuple = (
    #                         _score_tuple[0],
    #                         _score_tuple[1] + runtime.World_Graph.get_gdict()[neighbor['id']].get_population())
    #             if neighbor['id'] in node.broken_v:
    #                 continue
    #             #     _h = runtime.MAX
    #             # else:
    #             _h = heuristic_func(_score_tuple, agent_id)
    #             neighbor_node = GameNode(_agents_pos, _populated_v, _broken_v, _h, _id, _score_tuple, _parent)
    #             node.children.append(neighbor_node)
    #             ret_n = GameTree.minimax(neighbor_node, depth + 1,
    #                                      True, alpha, beta, heuristic_func, limit, agent_id)
    #             # If Semi-Cooperative and there's a tie
    #             if runtime.Active_Game_Type == "Semi-Cooperative" and ret_n.h == best.h:
    #                 if not best.score_tuple:
    #                     best = ret_n
    #                 else:
    #                     best = ret_n if sum(ret_n.score_tuple) >= sum(best.score_tuple) else best
    #             else:
    #                 if runtime.Active_Game_Type == "Fully-Cooperative":
    #                     best = ret_n if ret_n.h <= best.h else best
    #                 else:
    #                     best = ret_n if ret_n.h < best.h else best
    #             beta = min(beta, best.h)
    #
    #             # Alpha Beta Pruning
    #             if runtime.Pruning and beta <= alpha:
    #                 # print(f"pruned node:\n{neighbor_node}")
    #                 break
    #         if best == temp:
    #             return node
    #
    #         return best
    #

class GameNode:
    def __init__(self, agents_pos=None, populated_v=None, broken=None, h=None, _id=None, score_tuple=None, parent=None):
        self.id = _id
        self.agents_pos = agents_pos
        self.populated_v = populated_v
        self.broken_v = broken
        self.h = h
        self.parent = parent
        self.children = []
        self.score_tuple = score_tuple

    def __str__(self):
        return f"\tAgents Positions: {self.agents_pos},\n" \
               f"\tPopulated: {self.populated_v},\n" \
               f"\tBroken: {self.broken_v},\n" \
               f"\th: {self.h},\n" \
               f"\tScore Tuple: {self.score_tuple},\n" \
               f"\tParent: {self.parent}\n"
