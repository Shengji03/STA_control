"""A sampled configuration with its RRT parent and path cost."""

import copy
from src.geometry import Point


class Node:
    def __init__(self, point=None, cost=0.0, parent=-1):
        self.point = Point() if point is None else Point(point)
        self.cost = cost
        self.parent = parent

    def get_t(self):
        return self.point.get_t()

    def get_point(self):
        return copy.deepcopy(self.point)

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost

    def set_parent(self, parent):
        self.parent = parent
