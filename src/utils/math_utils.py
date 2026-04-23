import numpy as np
from src.constanst import MathConst


class MathUtils:
    @staticmethod
    def near_zero(value: float) -> bool:
        return np.abs(value) < MathConst.EPS
