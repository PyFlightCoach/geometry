from __future__ import annotations

from numbers import Number
from time import time
from typing import ClassVar, Literal, Self, TypeVar, overload
from warnings import warn

import numpy as np
import numpy.typing as npt

from geometry import Base


class TimeError(Exception):
    pass


T = TypeVar("T", float, npt.NDArray[np.float64])


class Time(Base):
    cols: ClassVar[list[str]] = ["t", "dt"]
    
    def __post_init__(self):
        self.t = self.t.astype(np.float64)
        self.dt = self.dt.astype(np.float64)
        if self.dt[-1] != 0:
            warn("Setting the last dt to 0, this is a new requirement")
            np.put(self.dt, -1, 0)

    @staticmethod
    def from_t(t: np.ndarray) -> Time:
        if isinstance(t, Number):
            return Time(t, 0)
        else:
            if len(t) == 1:
                dt = np.array([0])
            else:
                t = np.asarray(t, dtype=np.float64)
                arr = np.diff(t)
                dt = np.pad(arr, (0, 1))
            return Time(t, dt)

    @staticmethod
    def uniform(
        duration: float, npoints: int | None, minpoints: int = 1, freq=25
    ) -> Time:
        return Time.from_t(
            np.linspace(
                0,
                duration,
                npoints if npoints else max(int(np.ceil(duration * freq)), minpoints),
            )
        )

    @property
    def duration(self) -> float:
        return self.t[-1] - self.t[0]

    def scale(self, duration) -> Self:
        old_duration = self.t[-1] - self.t[0]
        sfac = duration / old_duration
        return Time(self.t[0] + (self.t - self.t[0]) * sfac, self.dt * sfac)

    def reset_zero(self):
        return Time(self.t - self.t[0], self.dt)

    @staticmethod
    def now():
        return Time.from_t(time())

    def extend(self, dt: float | None = None) -> Time:
        if dt is None:
            if len(self.dt) == 1:
                raise ValueError("Cannot extend time with no dt supplied if Time has only one point")
            dt = self.dt[-2]
        newdt = self.dt.copy()
        np.put(newdt, -1, dt)
        return Time(
            np.pad(self.t, (0, 1), constant_values=self.t[-1] + dt),
            np.pad(newdt, (0, 1), constant_values=0),
        )

    def linterp_recaclulate_dt(
        self,
        index: float | npt.NDArray | None = None,
        extrapolate: Literal["throw", "nearest"] = "throw",
    ):
        """linear interpolation"""
        index = np.arange(len(self)) if index is None else index
        interpolator = self.linterp(index, extrapolate)
        
        def dolinterp(new_index: npt.NDArray | Number):
            
            new_t = interpolator(new_index).t
            return Time.from_t(new_t)
#            next_index = np.searchsorted(extended_t.t, new_t, "right")
#
#            new_dt = extended_t.t[next_index] - new_t
#            return Time(new_t, new_dt)

        return dolinterp

    def __add__(self, t: float):
        return Time(self.t + t, self.dt)

    def __sub__(self, t: float):
        return Time(self.t - t, self.dt)

    @staticmethod
    def concatenate(times: list[Time]) -> Time:
        return Time.from_t(np.concatenate([t.t for t in times]))

    @overload
    def get_index(self, t: T, snap: Literal[False] = False) -> T: ...

    @overload
    def get_index(self, t: float, snap: Literal[True]) -> int: ...

    @overload
    def get_index(
        self, t: npt.NDArray[np.float64], snap: Literal[True]
    ) -> npt.NDArray[np.intp]: ...

    def get_index(
        self,
        t: float | npt.NDArray,
        snap: bool = False,
        extrapolate: Literal["throw", "nearest", "nan"] = "nan",
    ) -> int | float | npt.NDArray[np.integer | np.float64]:

        if extrapolate == "throw" and (np.any(t < self.t[0]) or np.any(t > self.t[-1])):
            raise ValueError(f"t={t} is out of bounds [{self.t[0]}, {self.t[-1]}]")

        if snap:
            if isinstance(t, Number):
                if extrapolate == "nan" and (t < self.t[0] or t > self.t[-1]):
                    return np.nan
                return np.argmin(np.abs(self.t - t))

            distances = np.abs(np.subtract.outer(t, self.t))
            indices = np.argmin(distances, axis=1)

            if extrapolate == "nan":
                out = indices.astype(np.float64)
                out[(t < self.t[0]) | (t > self.t[-1])] = np.nan
                return out

            return np.argmin(distances, axis=1)
        else:

            return np.interp(
                t,
                self.t,
                np.arange(len(self.t)),
                left=np.nan if extrapolate == "nan" else 0.0,
                right=np.nan if extrapolate == "nan" else float(len(self.t) - 1),
            )

    def get_value(self, index: Number | npt.NDArray) -> float | npt.NDArray:
        index = np.where(index < 0, len(self) + index, index)
        return np.interp(
            index,
            np.arange(len(self.t)),
            self.t,
            left=np.nan,
            right=np.nan,
        )