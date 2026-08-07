"""
Tests for Quaternion.squad.

ASSUMPTIONS TO VERIFY / ADJUST:
- Import paths below (geometry.quaternion / geometry.point / geometry.time)
  are guesses based on the relative imports seen in quaternion.py
  ("from .base import ..."). Fix these to match your actual package layout.
- Time(array_of_times) is assumed to be a valid constructor that exposes
  `.dt` (consecutive differences, length n-1) and a flat time array used
  for indexing inside squad. If Time's real constructor differs, adjust
  the `make_time()` helper only -- nothing else should need to change.
- Quaternion.almost_equal() is used for comparisons since it already
  handles the q / -q double-cover ambiguity via abs(dot).
"""
import numpy as np
import pytest

from geometry.base import ExtrapolationError
from geometry.point import Point
from geometry.quaternion import Quaternion
from geometry.time import Time


def make_time(ts: np.ndarray) -> Time:
    """Wrap raw time values in whatever Time expects. Adjust if Time's
    constructor signature differs."""
    return Time.from_t(ts)


def constant_rate_z_rotation(n: int, w: float, t_end: float):
    """Build n samples of a rotation purely about the z-axis, spinning at
    constant angular rate w (rad/s), plus the matching per-sample axis
    rates and timestamps. This is a geodesic (great-circle) path, which
    is a good sanity-check case for squad since the tangents are exact."""
    ts = np.linspace(0, t_end, n)
    angles = w * ts

    axang = Point(np.zeros(n), np.zeros(n), angles)
    r = Quaternion.from_axis_angle(axang)

    q_rate = Point(np.zeros(n), np.zeros(n), np.full(n, w))

    return r, q_rate, ts


class TestSquadBasics:
    def test_requires_at_least_two_samples(self):
        r, q, ts = constant_rate_z_rotation(1, 1.0, 1.0)
        t = make_time(ts)
        with pytest.raises(AssertionError):
            Quaternion.squad(r, q, t)

    def test_mismatched_lengths_raise(self):
        r, q, ts = constant_rate_z_rotation(5, 1.0, 4.0)
        t = make_time(ts)
        with pytest.raises(AssertionError):
            Quaternion.squad(r, Point(q.data[:-1]), t)

    def test_output_is_unit_quaternion(self):
        r, q, ts = constant_rate_z_rotation(5, np.pi / 3, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        query = np.linspace(ts[0], ts[-1], 25)
        out = interp(query)

        norms = abs(out)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-8)

    def test_extrapolation_raises(self):
        r, q, ts = constant_rate_z_rotation(5, np.pi / 4, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        with pytest.raises(ExtrapolationError):
            interp(np.array([ts[0] - 0.1]))

        with pytest.raises(ExtrapolationError):
            interp(np.array([ts[-1] + 0.1]))


class TestSquadPassesThroughKnots:
    def test_matches_samples_at_knot_times(self):
        r, q, ts = constant_rate_z_rotation(6, np.pi / 5, 5.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        out = interp(ts)
        assert out.almost_equal(r, tol=1e-6)


class TestSquadConstantOrientation:
    def test_zero_rate_same_orientation_is_constant(self):
        """If every sample is the same orientation with zero rate at every
        knot, squad should return that orientation everywhere (control
        points collapse to the sample itself)."""
        n = 4
        ts = np.linspace(0, 3, n)
        same_q = Quaternion.from_axis_angle(Point(0.3, -0.1, 0.2))
        r = Quaternion(np.tile(same_q.data, (n, 1)))
        q_rate = Point(np.zeros(n), np.zeros(n), np.zeros(n))

        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        query = np.linspace(ts[0], ts[-1], 10)
        out = interp(query)

        expected = Quaternion(np.tile(same_q.data, (len(query), 1)))
        assert out.almost_equal(expected, tol=1e-6)


class TestSquadAgainstAnalyticGeodesic:
    def test_constant_angular_rate_matches_analytic_rotation(self):
        """For pure single-axis rotation at constant angular rate, the
        path is a geodesic and squad (given exact tangents from the true
        rate) should reproduce the analytic rotation closely at
        off-knot times. This is a soft numerical check, not an exact
        equality -- loosen tol if it's too strict for your implementation."""
        w = np.pi / 2  # rad/s
        r, q_rate, ts = constant_rate_z_rotation(5, w, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        query_t = np.array([0.5, 1.5, 2.5, 3.5])
        out = interp(query_t)

        expected = Quaternion.from_axis_angle(
            Point(np.zeros(len(query_t)), np.zeros(len(query_t)), w * query_t)
        )

        assert out.almost_equal(expected, tol=1e-3)


class TestSquadContinuity:
    def test_no_jump_across_segment_boundary(self):
        """Sample densely across a knot and check consecutive outputs
        stay close together -- catches gross discontinuities/bugs in
        segment lookup at the boundary itself."""
        r, q, ts = constant_rate_z_rotation(5, np.pi / 6, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        knot = ts[2]
        around = np.linspace(knot - 0.05, knot + 0.05, 11)
        out = interp(around)

        # consecutive-sample dot products should stay near 1 (small angle
        # between neighbours) if there's no discontinuity at the knot
        dots = np.abs(out[:-1].dot(out[1:]))
        assert np.all(dots > 1 - 1e-3)