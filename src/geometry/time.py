from __future__ import annotations
from geometry import Base
from numbers import Number
from typing import Self, Literal
import numpy as np
import numpy.typing as npt
import pandas as pd
from time import time
from geometry.utils import get_index


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
    def uniform(duration: float, npoints: int | None, minpoints: int = 1) -> Time:
        return Time.from_t(
            np.linspace(
                0,
                duration,
                npoints if npoints else max(int(np.ceil(duration * 25)), minpoints),
            )
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

    def linterp(
        self,
        index: npt.NDArray | pd.Index,
        extrapolate: Literal["throw", "nearest"] = "throw",
    ):
        """linear interpolation between two times"""
        index = pd.Index(np.arange(len(self)) if index is None else index)
        assert len(index) == len(self)
        assert pd.Index(index).is_monotonic_increasing

        def dolinterp(ts: npt.NDArray | pd.Index):
            starts = index.get_indexer(ts, method="ffill")
            stops = index.get_indexer(ts, method="bfill")
            if np.any(starts * stops < 0) and extrapolate == "throw":
                raise Exception("Cannot extrapolate beyond parent range")

            new_time = Time.from_t(
                np.interp(ts, index, self.t, self.t[0], self.t[-1])
            )

            last_p = self[stops][-1]
            if last_p.t[0] == new_time.t[-1]:
                last_dt = last_p.dt[-1]
            else:
                last_dt = last_p.t[0] - new_time.t[-1]
            new_time.data[-1,-1] = last_dt    
            return new_time
        return dolinterp

    def interpolate_t(self, t: float):
        """get the floating point index at a given time"""
        return get_index(self.t, t)

    def __add__(self, t: float):
        return Time.from_t(self.t + t)
