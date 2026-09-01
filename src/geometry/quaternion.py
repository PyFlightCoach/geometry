"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Number
from typing import ClassVar, Literal
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd

from geometry.point import P0, PZ, Point

from .base import Base, ExtrapolationError, dprep

try:
    from scipy.interpolate import BSpline, UnivariateSpline, make_interp_spline
    from scipy.spatial.transform import Rotation, RotationSpline
    HAS_SCIPY = True
except ImportError:
    type UnivariateSpline=None
    type BSpline=None
    type Rotation=None
    type RotationSpline=None
    make_interp_spline=None
    HAS_SCIPY = False
    
def check_scipy():
    if not HAS_SCIPY:
        raise ImportError(
            "This function requires scipy. Please install scipy to use this feature."
        )


class Quaternion(Base):
    cols: ClassVar[list[str]] = ["w", "x", "y", "z"]

    @staticmethod
    def zero(count=1) -> Quaternion:
        return Quaternion(np.tile([1, 0, 0, 0], (count, 1)))

    @cached_property
    def xyzw(self):
        return self.data[:, [1, 2, 3, 0]]

    @property
    def axis(self) -> Point:
        return Point(self.data[:, 1:])

    @dprep
    def almost_equal(self, other: Quaternion, tol: float = 1e-6) -> bool:
        return np.all(np.abs(self.dot(other)) > 1 - tol)

    def positive(self) -> Quaternion:
        return Quaternion(np.where(self.data[:, 0] < 0, -self.data, self.data))

    def norm(self) -> Quaternion:
        return self / abs(self)

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        if hasattr(self, "_inverse"):
            return self._inverse
        self._inverse = self.conjugate().norm()
        return self._inverse

    def __mul__(self, other: Number | Quaternion | npt.NDArray) -> Quaternion:
        if isinstance(other, Quaternion):
            a, b = Quaternion.length_check(self, Quaternion.type_check(other))
            w = a.w * b.w - a.axis.dot(b.axis)
            xyz = a.w * b.axis + b.w * a.axis + a.axis.cross(b.axis)
            return Quaternion(np.column_stack([w, xyz.data]))

        elif isinstance(other, Number):
            return Quaternion(self.data * other)
        elif isinstance(other, np.ndarray):
            return Quaternion(self.data * self._dprep(other))

        raise TypeError(f"cant multiply a quaternion by a {other.__class__.__name__}")

    def __rmul__(self, other) -> Quaternion:
        # either it should have been picked up by the left hand object or it should commute
        return self * other

    def transform_point(self, point: Point) -> Point:
        """Transform a point by the rotation described by self"""
        a, b = Base.length_check(self, point)

        qdata = np.column_stack((np.zeros(len(a)), b.data))

        return (a * Quaternion(qdata) * a.inverse()).axis

    @staticmethod
    def from_euler(eul: Point) -> Quaternion:
        """Create a quaternion from a Point of Euler angles order z, y, x"""
        eul = Point.type_check(eul).unwrap()
        half = eul * 0.5
        c = half.cos
        s = half.sin

        return Quaternion(
            np.array(
                [
                    c.y * c.z * c.x + s.y * s.z * s.x,
                    c.y * c.z * s.x - s.y * s.z * c.x,
                    s.y * c.z * c.x + c.y * s.z * s.x,
                    c.y * s.z * c.x - s.y * c.z * s.x,
                ]
            ).T
        )

    def to_euler(self) -> Point:
        """Create a Point of Euler angles order z,y,x"""
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        with np.errstate(invalid="ignore"):
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        test = np.abs(sinp) >= 0.9999
        if len(sinp[test]) > 0:
            pitch[test] = np.copysign(np.pi / 2, sinp[test])
            yaw[test] = np.zeros(len(sinp[test]))

            roll[test] = 2 * np.arctan2(self.x[test], self.w[test])
        return Point(roll, pitch, yaw)

    @staticmethod
    def from_axis_angle(axangles: Point) -> Quaternion:
        angles = abs(axangles)

        c = np.cos(angles / 2)

        # sin(angle / 2) / angle, including the angle=0 limit
        scale = 0.5 * np.sinc(angles / (2 * np.pi))

        qdat = np.array([
            c,
            axangles.x * scale,
            axangles.y * scale,
            axangles.z * scale,
        ]).T

        return Quaternion(qdat)

    def to_axis_angle(self) -> Point:
        q = self.copy()

        replocs = q.w < 0
        q.data[replocs, :] *= -1

        return q._to_axis_angle()

    def _to_axis_angle(self) -> Point:
        """To a point of axis angles. Must be normalized first."""

        w = np.clip(self.w, -1.0, 1.0)

        angle = 2 * np.arccos(w)
        s = np.sqrt(1 - w**2)

        sangle = np.divide(
            angle,
            s,
            out=np.full_like(angle, 2.0),
            where=s >= 1e-6,
        )

        return self.axis * sangle

    @staticmethod
    def axis_rates(q: Quaternion, qdot: Quaternion) -> Point:
        wdash = qdot * q.conjugate()
        return wdash.norm().to_axis_angle()

    @staticmethod
    def _axis_rates(q: Quaternion, qdot: Quaternion) -> Point:
        wdash = qdot * q.conjugate()
        return wdash.norm()._to_axis_angle()

    @staticmethod
    def body_axis_rates(q: Quaternion, qdot: Quaternion) -> Point:
        wdash = q.conjugate() * qdot
        return wdash.norm().to_axis_angle()

    @staticmethod
    def _body_axis_rates(q: Quaternion, qdot: Quaternion) -> Point:
        wdash = q.conjugate() * qdot
        return wdash.norm()._to_axis_angle()

    def rotate(self, rate: Point) -> Quaternion:
        return (Quaternion.from_axis_angle(rate) * self).norm()

    def body_rotate(self, rate: Point) -> Quaternion:
        return (self * Quaternion.from_axis_angle(rate)).norm()

    def diff(
        self, dt: Number | npt.NDArray = None, mode: Literal["world", "body"] = "world"
    ) -> Point:
        """differentiate in the world frame"""
        if len(self) == 1:
            return P0()
        if not pd.api.types.is_list_like(dt):
            dt = np.full(len(self), 1 if not dt else dt)
        assert len(dt) == len(self)
        dt = dt * len(dt) / (len(dt) - 1)

        method = (
            Quaternion._axis_rates if mode == "world" else Quaternion._body_axis_rates
        )

        ps = (
            method(Quaternion(self.data[:-1, :]), Quaternion(self.data[1:, :]))
            / dt[:-1]
        )

        return Point(np.pad(ps.data, ((0, 1), (0, 0)), mode="edge"))

    def body_diff(self, dt: Number | npt.NDArray = None) -> Point:
        return self.diff(dt, "body")

    def to_rotation_matrix(self) -> npt.NDArray[np.float64]:
        """http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        https://github.com/mortlind/pymath3d/blob/master/math3d/quaternion.py
        """
        n = self.norm()
        s, x, y, z = n.w, n.x, n.y, n.z
        x2, y2, z2 = n.x**2, n.y**2, n.z**2
        return np.array(
            [
                [1 - 2 * (y2 + z2), 2 * x * y - 2 * s * z, 2 * s * y + 2 * x * z],
                [2 * x * y + 2 * s * z, 1 - 2 * (x2 + z2), -2 * s * x + 2 * y * z],
                [-2 * s * y + 2 * x * z, 2 * s * x + 2 * y * z, 1 - 2 * (x2 + y2)],
            ]
        ).T

    @staticmethod
    def from_rotation_matrix(matrix: npt.NDArray[np.float64]) -> Quaternion:
        # This method assumes row-vector and postmultiplication of that vector
        m = matrix.conj().transpose()
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

        q = np.array(q).astype("float64")
        q *= 0.5 / np.sqrt(t)
        return Quaternion(*q)

    def closest_principal(self) -> Quaternion:
        eul = self.to_euler()
        rads = eul * (2 / np.pi)
        return Quaternion.from_euler(rads.round(0) * np.pi / 2)

    def is_inverted(self) -> bool:
        # does the rotation reverse the Z axis?
        return np.sign(self.transform_point(PZ()).z) > 0

    def bearing(self, p: Point = None):
        if p is None:
            p = Point.X()
        return self.transform_point(p).bearing()


    def bounded_by(self, tol: float):
        """Check all rotations within this dataset are within tol radians of the first one"""

        return len(self) == 1 or np.all(
            [abs(Quaternion.body_axis_rates(self[1:], self[0])) < tol]
        )

    def plot_3d(self, size: float = 3, vis: Literal["coord", "plane"] = "coord"):
        from geometry import Transformation

        return Transformation(self).plot_3d(size, vis)


    @staticmethod
    def slerp_pair(
        q0: Quaternion,
        q1: Quaternion,
        frac: npt.NDArray,
        dt: npt.NDArray | Number = 1,
        shortest: bool = True,
        mode: Literal["world", "body"] = "world",
    ) -> tuple[Quaternion, Point]:
        """Batched spherical linear interpolation between corresponding
        rows of q0 and q1. frac in [0, 1], broadcastable to len(q0)."""
        q0, q1 = Quaternion.length_check(q0, q1)

        frac = np.asarray(frac)

        q0_data = q0.data
        q1_data = q1.data.copy()

        dot = np.sum(q0_data * q1_data, axis=1)

        if shortest:
            flip = dot < 0
            q1_data = np.where(flip[:, None], -q1_data, q1_data)
            dot = np.where(flip, -dot, dot)

        dot = np.clip(dot, -1.0, 1.0)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        small = sin_theta_0 < 1e-6
        safe_sin = np.where(small, 1.0, sin_theta_0)

        s0 = np.where(small, 1.0 - frac, np.sin((1.0 - frac) * theta_0) / safe_sin)
        s1 = np.where(small, frac, np.sin(frac * theta_0) / safe_sin)

        out = Quaternion(s0[:, None] * q0_data + s1[:, None] * q1_data).norm()

        if mode == "world":
            relative = Quaternion(q1_data) * q0.inverse()
        else:
            relative = q0.inverse() * Quaternion(q1_data)
        rate = relative.to_axis_angle() / dt

        return out, rate

    def slerp(
        self,
        index: npt.NDArray = None,
    ):
        index = np.asarray(np.arange(len(self)) if index is None else index)

        assert len(index) == len(self)
        assert np.all(index[1:] >= index[:-1])

        return SlerpFunction(index, self)

    def squad(self: Quaternion, q: Point, index: npt.NDArray[np.float64]) -> callable:
        assert len(self) == len(q) == len(index)
        assert len(self) >= 2, "squad needs at least two samples"       
        assert np.all(index[1:] >= index[:-1])

        return SquadFunction.build(index, self, q)


    def rotation_spline(
        self, index: npt.NDArray[np.float64], **kwargs
    ) -> Callable[[npt.NDArray[np.float64]], Quaternion | Point]:
        
        assert len(self) == len(index)
        assert len(self) >= 2, "squad needs at least two samples"
        assert np.all(index[1:] >= index[:-1])

        return RotationSplineFunction.build(index, self)


def Q0(count=1):
    return Quaternion.zero(count)


def Quaternions(*args, **kwargs):
    warn(
        "Quaternions is deprecated, you can now just use Quaternion", DeprecationWarning
    )
    return Quaternions(*args, **kwargs)


@dataclass
class SlerpFunction:
    index: npt.NDArray[np.float64]
    data: Quaternion

    def __call__(
        self,
        ts: npt.NDArray | Number,
        mode: Literal["world", "body"] = "world",
        axis_rates: bool = False,
        extrapolate: Literal["throw", "nearest", "nan"] = "nearest",
    ) -> tuple[Quaternion, Point] | Quaternion:
        ts = np.atleast_1d(ts)
        starts = np.searchsorted(self.index, ts, side="right") - 1
        stops = np.searchsorted(self.index, ts, side="left")

        odata = self.data[starts].data
        rdata = np.zeros((len(ts), 3))
        to_interp = starts != stops

        q, rate = Quaternion.slerp_pair(
            self.data[starts[to_interp]],
            self.data[stops[to_interp]],
            (ts[to_interp] - self.index[starts[to_interp]])
            / (self.index[stops[to_interp]] - self.index[starts[to_interp]]),
            self.index[stops[to_interp]] - self.index[starts[to_interp]],
            True,
            mode,
        )

        odata[to_interp] = q.data
        rdata[to_interp] = rate.data
        # case exact match (start == stop): rdata stays zero — instantaneous
        # rate at a sample point isn't defined by a pairwise slerp; use
        # dosquad-style central differencing there if you need it.

        aboves = stops == -1
        belows = starts == -1

        if np.any(aboves) or np.any(belows):
            if extrapolate == "throw":
                raise ExtrapolationError("Cannot slerp beyond range")
            elif extrapolate == "nan":
                odata[aboves & belows] = np.nan
                rdata[aboves & belows] = np.nan
            elif extrapolate == "nearest":
                odata[aboves] = self.data.data[-1, :]
                odata[belows] = self.data.data[0, :]
                rdata[aboves & belows] = 0.0

        if axis_rates:
            return Quaternion(odata), Point(rdata)
        else:
            return Quaternion(odata)

    def to_dict(self) -> dict:
        return {"index": self.index.tolist(), "data": self.data.to_dict()}

    @staticmethod
    def from_dict(data: dict) -> SlerpFunction:
        return SlerpFunction(
            np.asarray(data["index"]), Quaternion.from_dict(data["data"])
        )


@dataclass
class SquadFunction:
    index: npt.NDArray[np.float64]
    data: Quaternion
    q: Point
    s0: Quaternion
    s1: Quaternion

    @staticmethod
    def build(
        index: npt.NDArray[np.float64], data: Quaternion, q: Point
    ) -> SquadFunction:
        dt = np.pad(np.diff(index), (0, 1))
        s0, s1 = SquadFunction.squad_control_points(
            data[:-1], data[1:], q[:-1], q[1:], dt[:-1]
        )
        return SquadFunction(index, data, q, s0, s1)

    @staticmethod
    def squad_control_points(
        r0: Quaternion, r1: Quaternion, q0: Point, q1: Point, dt: float
    ) -> tuple[Quaternion, Quaternion]:
        r1 = r1.where(Quaternion.dot(r0, r1) < 0, -r1)

        exp_cur = Quaternion.from_axis_angle(q0 * dt / 3)
        exp_next = Quaternion.from_axis_angle(-q1 * dt / 3)

        return r0 * exp_cur, r1 * exp_next

    def _squad_quat(
        self, ts: npt.NDArray, mode: Literal["world", "body"] = "world"
    ) -> Quaternion:
        starts = np.searchsorted(self.index, ts, side="right") - 1
        starts = np.clip(starts, 0, len(self.index) - 2)

        seg_dt = self.index[starts + 1] - self.index[starts]
        u = np.where(seg_dt > 0, (ts - self.index[starts]) / seg_dt, 0.0)

        P0, P1, P2, P3 = (
            self[starts],
            self.s0[starts],
            self.s1[starts],
            self[starts + 1],
        )

        A, _ = Quaternion.slerp_pair(P0, P1, u, seg_dt, True, mode)
        B, _ = Quaternion.slerp_pair(P1, P2, u, seg_dt, True, mode)
        C, _ = Quaternion.slerp_pair(P2, P3, u, seg_dt, True, mode)

        D, _ = Quaternion.slerp_pair(A, B, u, seg_dt, True, mode)
        E, _ = Quaternion.slerp_pair(B, C, u, seg_dt, True, mode)

        out, _ = Quaternion.slerp_pair(D, E, u, seg_dt, True, mode)
        return out

    def __call__(
        self, ts: npt.NDArray | Number, mode: Literal["world", "body"] = "world"
    ) -> tuple[Quaternion, Point]:
        ts = np.atleast_1d(np.asarray(ts, dtype=float))

        if np.any(ts < self.index[0]) or np.any(ts > self.index[-1]):
            raise ExtrapolationError("Cannot squad beyond range")

        out = self._squad_quat(ts, mode)

        span = self.index[-1] - self.index[0]
        eps = max(span * 1e-6, 1e-9)
        ts_p = np.minimum(ts + eps, self.index[-1])
        ts_m = np.maximum(ts - eps, self.index[0])
        dt_eff = ts_p - ts_m
        dt_eff = np.where(dt_eff > 0, dt_eff, eps)

        q_m = self._squad_quat(ts_m, mode)
        q_p = self._squad_quat(ts_p, mode)

        method = (
            Quaternion._axis_rates if mode == "world" else Quaternion._body_axis_rates
        )
        rate = method(q_m, q_p) / dt_eff

        return out, rate

    def to_dict(self) -> dict:
        return {
            "index": self.index.tolist(),
            "data": self.data.to_dict(),
            "q": self.q.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> SquadFunction:
        return SquadFunction.build(
            np.asarray(data["index"]),
            Quaternion.from_dict(data["data"]),
            Point.from_dict(data["q"]),
        )


@dataclass
class RotationSplineFunction:
    index: npt.NDArray[np.float64]
    data: Quaternion
    spline: RotationSpline

    def __call__(self, x: npt.NDArray[np.float64], n=0) -> Quaternion:
        if n == 0:
            return Quaternion(self.spline(x).as_quat()[:, [3, 0, 1, 2]])
        else:
            return Point(self.spline(x, n))

    def to_dict(self) -> dict:
        return {
            "index": self.index.tolist(),
            "data": self.data.to_dict(),
        }

    @staticmethod
    def build(index: npt.NDArray[np.float64], data: Quaternion) -> RotationSplineFunction:
        check_scipy()
        return RotationSplineFunction(
            index,
            data,
            RotationSpline(index, Rotation.from_quat(data.data[:, [1, 2, 3, 0]])),
        )

    @staticmethod
    def from_dict(data: dict) -> RotationSplineFunction:
        check_scipy()
        return RotationSplineFunction.build(
            np.asarray(data["index"]),
            Quaternion.from_dict(data["data"]),
        )
