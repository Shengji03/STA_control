import copy
from typing import overload, Union, Iterable, List

import numpy as np

from .geometry import Geometry
from .simplex import Simplex
from .interface import Support


class Point(Geometry, Simplex, Support):

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, t: Union[np.ndarray, Iterable, int, float]) -> None:
        ...

    @overload
    def __init__(self, t: Geometry) -> None:
        ...

    def __init__(self, t=None) -> None:
        super().__init__()
        if t is None:
            self.t = np.zeros(1)
        elif isinstance(t, (np.ndarray, Iterable, int, float)):
            if isinstance(t[0], Point):
                self.t = np.array(t[0].get_t(), dtype=np.float64)
            else:
                self.t = np.array(t, dtype=np.float64)
        elif isinstance(t, Point):
            self.t = np.array(t.get_t())
        else:
            raise ValueError("bad argument to constructor")
        self.dim: int = self.t.size

    def get_tx(self) -> float:
        return self.t[0]

    def get_ty(self) -> float:
        return self.t[1]

    def get_tz(self) -> float:
        return self.t[2]

    def set_tx(self, num: float) -> None:
        self.t[0] = num

    def set_ty(self, num: float) -> None:
        self.t[1] = num

    def set_tz(self, num: float) -> None:
        self.t[2] = num

    @property
    def points(self):
        return [copy.deepcopy(self)]

    def calculate_closest_point_to_origin(self) -> Geometry:
        return copy.deepcopy(self)

    def calculate_barycentric_coordinates(self, geometry: Geometry) -> List[float]:
        return [1.0]

    def plot(self, ax, c=None):
        ax.scatter(*self.t, c='r' if c is None else c)
