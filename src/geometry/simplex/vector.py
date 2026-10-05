from typing import overload, Union, Iterable

import numpy as np

from .geometry import Geometry
from .point import Point


class Vector(Geometry):

    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, t0: Union[np.ndarray, Iterable, int, float]):
        ...

    @overload
    def __init__(self, t0: Union[np.ndarray, Iterable, int, float], t1: Union[np.ndarray, Iterable, int, float]):
        ...

    @overload
    def __init__(self, t0: Point):
        ...

    @overload
    def __init__(self, t0: Point, t1: Point):
        ...

    def __init__(self, t0=None, t1=None) -> None:
        super().__init__()
        if t0 is None:
            self.t = np.zeros(1)
        elif isinstance(t0, (np.ndarray, Iterable, int, float)):
            if t1 is None:
                self.t = np.array(t0, dtype=np.float64)
            elif isinstance(t1, (np.ndarray, Iterable, int, float)):
                self.t = np.array(t1, dtype=np.float64) - np.array(t0, dtype=np.float64)
        elif isinstance(t0, Point):
            if t1 is None:
                self.t = t0.get_t()
            elif isinstance(t1, Point):
                self.t = t1.get_t() - t0.get_t()

        else:
            raise ValueError("bad argument to constructor")

    def __add__(self, other: Geometry):
        return Vector(np.zeros_like(self.t), self.t + other.get_t())

    def __sub__(self, other: Geometry):
        return Vector(np.zeros_like(self.t), self.t - other.get_t())

    def __mul__(self, other):
        return Vector(np.zeros_like(self.t), other * self.t)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vector(np.zeros_like(self.t), self.t / other)

    def __neg__(self):
        return self.__class__(-self.t)

    def norm(self) -> float:
        return np.linalg.norm(self.t)
