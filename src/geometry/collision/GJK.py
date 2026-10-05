"""GJK intersection check for the convex geometries used by robot RRT."""

import copy
import numpy as np

from src.constanst import MathConst
from ..simplex import Point, LineSegment, Triangle, Tetrahedron, UnitVector, Support


def _make_simplex(points):
    constructors = {1: Point, 2: LineSegment, 3: Triangle, 4: Tetrahedron}
    return constructors[len(points)](copy.deepcopy(points))


class GJK:
    @staticmethod
    def is_intersecting(shape0: Support, shape1: Support):
        origin = Point([0, 0, 0])
        unit_vector = UnitVector(np.array([1, 0, 0]))

        point = shape0.calculate_support_point(unit_vector) - shape1.calculate_support_point(-unit_vector)
        points = [point]
        closest_point = point

        coordinates = [1]

        while closest_point != origin:
            unit_vector = -UnitVector(closest_point)
            point = shape0.calculate_support_point(unit_vector) - shape1.calculate_support_point(-unit_vector)
            if np.dot(point.get_t(), unit_vector.get_t()) < 0:
                return False
            points.append(point)

            if len(points) == 5:
                coordinate_min = min(coordinates)
                coordinate_min_index = coordinates.index(coordinate_min)
                points.pop(coordinate_min_index)
                break

            simplex = _make_simplex(points)
            closest_point = simplex.calculate_closest_point_to_origin()
            coordinates = simplex.calculate_barycentric_coordinates(closest_point)

            j = 0
            for i, coordinate in enumerate(coordinates):
                if abs(coordinate) < MathConst.EPS:
                    points.pop(i - j)
                    j = j + 1

        return True
