from ..simplex import Support
from .GJK import GJK


class Collision:
    """Check intersections between convex support geometries."""

    @staticmethod
    def is_collision(shape0: Support, shape1: Support) -> bool:
        return GJK.is_intersecting(shape0, shape1)
