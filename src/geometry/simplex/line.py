import copy
from typing import overload, Union, Iterable, List, Tuple

import numpy as np

from .point import Point


class Line:
    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, point0: Point, point1: Point) -> None:
        ...

    @overload
    def __init__(self, point0: Union[np.ndarray, Iterable, int, float],
                 point1: Union[np.ndarray, Iterable, int, float]) -> None:
        ...

    @overload
    def __init__(self, point0: List[Point], point1: None) -> None:
        ...

    def __init__(self, point0=None, point1=None) -> None:
        super().__init__()
        if (point0 is None) and (point1 is None):
            self.point0 = Point()
            self.point1 = Point()
        elif isinstance(point0, (List, Tuple)) and (point1 is None):
            self.point0 = point0[0]
            self.point1 = point0[1]
        elif isinstance(point0, (Point, np.ndarray, Iterable, int, float)) \
                and isinstance(point1, (Point, np.ndarray, Iterable, int, float)):
            self.point0 = Point(point0)
            self.point1 = Point(point1)
        else:
            raise ValueError("bad argument to constructor")
        self.length = np.linalg.norm((self.point1 - self.point0).get_t())

    def get_point0(self) -> Point:
        return copy.deepcopy(self.point0)

    def get_point1(self) -> Point:
        return copy.deepcopy(self.point1)
