"""Point distance used by the retained RRT planner."""

import numpy as np

from ..simplex import Point


class Distance:
    @staticmethod
    def point_to_point(point0: Point, point1: Point):
        return np.linalg.norm((point1 - point0).get_t())
