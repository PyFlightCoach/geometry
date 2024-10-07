from __future__ import annotations
from geometry import Base
from numbers import Number
from typing import Self
import numpy as np
from time import time


class Time(Base):
    cols = ["t", "dt"]

    @staticmethod
    def from_t(t: np.ndarray) -> Time:
        if isinstance(t, Number):
            return Time(t, 1 / 25)
        else:
            if len(t) == 1:
                dt = np.array([1 / 25])
            else:
                arr = np.diff(t)
                dt = np.concatenate([arr, [arr[-1]]])
            return Time(t, dt)

    @staticmethod
    def uniform(duration: float, npoints: int | None, minpoints:int=1) -> Time:
        return Time.from_t(
            np.linspace(0, duration, npoints if npoints else max(int(np.ceil(duration * 25)), minpoints))
        )

    def scale(self, duration) -> Self:
        old_duration = self.t[-1] - self.t[0]
        sfac = duration / old_duration
        return Time(self.t[0] + (self.t - self.t[0]) * sfac, self.dt * sfac)

    def reset_zero(self):
        return Time(self.t - self.t[0], self.dt)

    @staticmethod
    def now():
        return Time.from_t(time())

    def extend(self):
        return Time.concatenate([self, Time(self.t[-1] + self.dt[-1], self.dt[-1])])
