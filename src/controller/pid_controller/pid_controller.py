from ..controller import Controller


class PIDController(Controller):
    """
    backward euler and use filtered derivative
    """

    def __init__(self, kp: float, ki: float, kd: float, ts=0.001, filter_coefficient=100.0):
        self.__kp = kp
        self.__ki = ki
        self.__kd = kd
        self.__filter_coefficient = filter_coefficient

        self.__prev_input = 0.0
        self.__integral = 0.0
        self.__ts = ts
        self.__prev_derivative_out = 0.0

    def control(self, inp):
        self.__integral += inp * self.__ts

        # derivative = (inp - self.__prev_input) / self.__ts
        derivative = (self.__filter_coefficient * (inp - self.__prev_input) + self.__prev_derivative_out) / (
                    self.__filter_coefficient * self.__ts + 1.0)
        self.__prev_derivative_out = derivative
        output = self.__kp * inp + self.__ki * self.__integral + self.__kd * derivative

        self.__prev_input = inp

        return output

    def reset(self):
        self.__prev_input = 0.0
        self.__integral = 0.0

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, ts):
        self.__ts = ts

    @property
    def kp(self):
        return self.__kp

    @property
    def ki(self):
        return self.__ki

    @property
    def kd(self):
        return self.__kd

    @kp.setter
    def kp(self, kp):
        self.__kp = kp

    @ki.setter
    def ki(self, ki):
        self.__ki = ki

    @kd.setter
    def kd(self, kd):
        self.__kd = kd

    def set_parameter(self, kp: float, ki: float, kd: float):
        self.__kp = kp
        self.__ki = ki
        self.__kd = kd
