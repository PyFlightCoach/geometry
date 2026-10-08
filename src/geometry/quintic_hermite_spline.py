from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from geometry import Point


@dataclass
class QuinticHermiteSpline:
    time: npt.NDArray[np.float64]
    pos: Point
    vel: Point
    acc: Point

    def __post_init__(self):
        n = len(self.time)
        assert len(self.pos) == n

        assert len(self.pos) == n
        assert len(self.vel) == n, (
                "Velocity array length must match time and position arrays"
            )
        assert len(self.acc) == n, (
                "Acceleration array length must match time and position arrays"
            )
    
    def __call__(
        self, t_query: npt.NDArray[np.float64] | float
    ) -> tuple[Point, Point, Point]:
        """Evaluate position, velocity, and acceleration at query times."""
        idx, u, dt = self._normalized_time(t_query)
        boundary_constraints = self._boundary_constraints(idx)

        pos = self._quintic_position(u, dt, *boundary_constraints)
        vel = self._quintic_velocity(u, dt, *boundary_constraints)
        acc = self._quintic_acceleration(u, dt, *boundary_constraints)

        return pos, vel, acc

    def pos_spline(self, t: npt.NDArray[np.float64]) -> Point:
        idx, u, dt = self._normalized_time(t)
        boundary_constraints = self._boundary_constraints(idx)
       
        pos = self._quintic_position(u, dt, *boundary_constraints)
        return pos

    def vel_spline(self, t: npt.NDArray[np.float64]) -> Point:
        idx, u, dt = self._normalized_time(t)
        boundary_constraints = self._boundary_constraints(idx)

        vel = self._quintic_velocity(u, dt, *boundary_constraints)
        return vel

    def acc_spline(self, t: npt.NDArray[np.float64]) -> Point:
        idx, u, dt = self._normalized_time(t)
        boundary_constraints = self._boundary_constraints(idx)

        acc = self._quintic_acceleration(u, dt, *boundary_constraints)
        return acc

    def _normalized_time(self, t_query):
        t_q = np.atleast_1d(t_query).astype(float)
        t_clipped = np.clip(t_q, self.time[0], self.time[-1])

        idx = np.searchsorted(self.time, t_clipped)
        idx = np.clip(idx, 1, len(self.time) - 1)

        t0 = self.time[idx - 1]
        t1 = self.time[idx]
        dt = np.where(t1 - t0 == 0.0, 1e-9, t1 - t0)
        u = (t_clipped - t0) / dt

        return idx, u, dt

    def _boundary_constraints(self, idx):
        """Extracts the boundary constraints for the active segments."""
        p0, p1 = self.pos[idx - 1], self.pos[idx]
        v0, v1 = self.vel[idx - 1], self.vel[idx]
        a0, a1 = self.acc[idx - 1], self.acc[idx]
        return p0, p1, v0, v1, a0, a1

    def _cubic_position(self, u, dt, p0, p1, v0, v1):
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2

        return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1

    def _quintic_position(self, u, dt, p0, p1, v0, v1, a0, a1):
        """Evaluates 3D position using base quintic polynomials."""
        h0 = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
        h1 = u - 6 * u**3 + 8 * u**4 - 3 * u**5
        h2 = 0.5 * u**2 - 1.5 * u**3 + 1.5 * u**4 - 0.5 * u**5
        h3 = 0.5 * u**3 - u**4 + 0.5 * u**5
        h4 = -4 * u**3 + 7 * u**4 - 3 * u**5
        h5 = 10 * u**3 - 15 * u**4 + 6 * u**5

        return (
            h0 * p0
            + h1 * (v0 * dt)
            + h2 * (a0 * dt**2)
            + h3 * (a1 * dt**2)
            + h4 * (v1 * dt)
            + h5 * p1
        )

    def _cubic_velocity(self, u, dt, p0, p1, v0, v1):
        dh00 = 6 * u**2 - 6 * u
        dh10 = 3 * u**2 - 4 * u + 1
        dh01 = -6 * u**2 + 6 * u
        dh11 = 3 * u**2 - 2 * u

        return (dh00 * p0 + dh10 * dt * v0 + dh01 * p1 + dh11 * dt * v1) / dt

    def _quintic_velocity(self, u, dt, p0, p1, v0, v1, a0, a1):
        """Evaluates 3D velocity using first-derivative polynomials."""
        dh0 = -30 * u**2 + 60 * u**3 - 30 * u**4
        dh1 = 1 - 18 * u**2 + 32 * u**3 - 15 * u**4
        dh2 = u - 4.5 * u**2 + 6 * u**3 - 2.5 * u**4
        dh3 = 1.5 * u**2 - 4 * u**3 + 2.5 * u**4
        dh4 = -12 * u**2 + 28 * u**3 - 15 * u**4
        dh5 = 30 * u**2 - 60 * u**3 + 30 * u**4

        return (
            dh0 * p0
            + dh1 * (v0 * dt)
            + dh2 * (a0 * dt**2)
            + dh3 * (a1 * dt**2)
            + dh4 * (v1 * dt)
            + dh5 * p1
        ) / dt

    def _cubic_acceleration(self, u, dt, p0, p1, v0, v1):
        d2h00 = 12 * u - 6
        d2h10 = 6 * u - 4
        d2h01 = -12 * u + 6
        d2h11 = 6 * u - 2

        return (d2h00 * p0 + d2h10 * dt * v0 + d2h01 * p1 + d2h11 * dt * v1) / dt**2

    def _quintic_acceleration(self, u, dt, p0, p1, v0, v1, a0, a1):
        """Evaluates 3D acceleration using second-derivative polynomials."""
        d2h0 = -60 * u + 180 * u**2 - 120 * u**3
        d2h1 = -36 * u + 96 * u**2 - 60 * u**3
        d2h2 = 1 - 9 * u + 18 * u**2 - 10 * u**3
        d2h3 = 3 * u - 12 * u**2 + 10 * u**3
        d2h4 = -24 * u + 84 * u**2 - 60 * u**3
        d2h5 = 60 * u - 180 * u**2 + 120 * u**3

        return (
            d2h0 * p0
            + d2h1 * (v0 * dt)
            + d2h2 * (a0 * dt**2)
            + d2h3 * (a1 * dt**2)
            + d2h4 * (v1 * dt)
            + d2h5 * p1
        ) / (dt**2)
