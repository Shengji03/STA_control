"""Box support geometry for the carried shade board and obstacles."""

import numpy as np
from spatialmath import SE3

from ..simplex import Point, Support
from .geometry3d import Geometry3D


class Brick(Geometry3D, Support):
    def __init__(self, base: SE3, dimensions: np.ndarray) -> None:
        super().__init__(base)
        self.__dimensions = np.asarray(dimensions).copy()

    @property
    def dimensions(self):
        return self.__dimensions.copy()

    @property
    def points(self):
        return [
            Point((self.base * SE3.Trans(
                *(self.__dimensions * np.array([
                    i % 2 - 0.5, i % 4 // 2 - 0.5, i // 4 - 0.5,
                ]))
            )).t)
            for i in range(8)
        ]

    def plot(self, ax, c=None):
        vertices = np.array([point.get_t() for point in self.points])
        faces = (
            (0, 2, 4, 6), (1, 3, 5, 7),
            (0, 1, 4, 5), (2, 3, 6, 7),
            (0, 1, 2, 3), (4, 5, 6, 7),
        )
        for face in faces:
            coordinates = vertices[list(face)].reshape(2, 2, 3)
            ax.plot_surface(
                coordinates[:, :, 0], coordinates[:, :, 1], coordinates[:, :, 2],
                alpha=0.5, color='b' if c is None else c,
            )
