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
        out, rate = interp(query)

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

        out, rate = interp(ts)
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
        out, rate = interp(query)

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
        out, rate = interp(query_t)

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
        out, rate = interp(around)

        # consecutive-sample dot products should stay near 1 (small angle
        # between neighbours) if there's no discontinuity at the knot
        dots = np.abs(out[:-1].dot(out[1:]))
        assert np.all(dots > 1 - 1e-3)


# ---------------------------------------------------------------------------
# New: coverage for the axis-rate return value added to dosquad.
# ---------------------------------------------------------------------------


class TestSquadRateBasics:
    def test_rate_is_a_point_matching_query_length(self):
        r, q, ts = constant_rate_z_rotation(5, np.pi / 3, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        query = np.linspace(ts[0], ts[-1], 13)
        out, rate = interp(query)

        assert isinstance(rate, Point)
        assert len(rate) == len(query)

    def test_rate_has_no_nans_including_at_knots_and_endpoints(self):
        """The central-difference eps clamps to the domain edge at ts[0]
        and ts[-1], which is where a naive implementation is most likely
        to divide by zero or produce nan/inf."""
        r, q, ts = constant_rate_z_rotation(6, np.pi / 5, 5.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        out, rate = interp(ts)  # exact knots, including both endpoints
        assert np.all(np.isfinite(rate.data))

    def test_default_mode_is_world_and_body_mode_is_accepted(self):
        r, q, ts = constant_rate_z_rotation(5, np.pi / 4, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q, t)

        query = np.linspace(ts[0], ts[-1], 9)
        out_default, rate_default = interp(query)
        out_world, rate_world = interp(query, "world")
        out_body, rate_body = interp(query, "body")

        assert out_default.almost_equal(out_world, tol=1e-12)
        np.testing.assert_allclose(rate_default.data, rate_world.data, atol=1e-12)
        # orientation output shouldn't depend on which frame the rate is
        # reported in
        assert out_world.almost_equal(out_body, tol=1e-8)


class TestSquadRateAgainstAnalytic:
    def test_constant_angular_rate_recovers_true_rate(self):
        """For a pure single-axis rotation at constant angular rate, the
        Hermite tangents fed into squad_control_points are already
        consistent with a single geodesic, so the exact spherical cubic
        Bezier (3-level de Casteljau) degenerates to plain constant-speed
        slerp everywhere in the segment -- not just at the knots. The
        central-differenced rate should therefore recover w on the
        z-axis and ~0 elsewhere at ANY query point, including
        mid-segment. (Previously this only held approximately, with a
        systematic 5/4 * w bias at segment midpoints under the old
        Shoemake-shortcut + dt/4 control points -- see git history /
        PR discussion for the derivation.)"""
        w = np.pi / 2
        r, q_rate, ts = constant_rate_z_rotation(5, w, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        query_t = np.array([0.5, 1.5, 2.5, 3.5])
        out, rate = interp(query_t)

        expected_rate = Point(
            np.zeros(len(query_t)), np.zeros(len(query_t)), np.full(len(query_t), w)
        )

        np.testing.assert_allclose(rate.x, expected_rate.x, atol=1e-3)
        np.testing.assert_allclose(rate.y, expected_rate.y, atol=1e-3)
        np.testing.assert_allclose(rate.z, expected_rate.z, atol=1e-3)

    def test_constant_angular_rate_matches_at_dense_non_midpoint_queries(self):
        """Same as above but at points that are NOT segment midpoints,
        to make sure the exact match isn't an artifact of the u=0.5
        symmetry point specifically."""
        w = np.pi / 3
        r, q_rate, ts = constant_rate_z_rotation(6, w, 5.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        query_t = np.linspace(ts[0], ts[-1], 37)[1:-1]  # avoid exact knots
        out, rate = interp(query_t)

        np.testing.assert_allclose(rate.z, np.full(len(query_t), w), atol=1e-3)
        np.testing.assert_allclose(rate.x, np.zeros(len(query_t)), atol=1e-3)
        np.testing.assert_allclose(rate.y, np.zeros(len(query_t)), atol=1e-3)

    def test_world_and_body_rate_agree_for_single_axis_rotation(self):
        """A rotation confined to a single fixed axis commutes with
        itself, so the body-frame and world-frame angular rate should be
        identical (only genuinely multi-axis motion should make them
        diverge)."""
        w = np.pi / 3
        r, q_rate, ts = constant_rate_z_rotation(6, w, 5.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        query_t = np.linspace(ts[0], ts[-1], 11)
        _, rate_world = interp(query_t, "world")
        _, rate_body = interp(query_t, "body")

        np.testing.assert_allclose(rate_world.data, rate_body.data, atol=1e-6)


class TestSquadRateConstantOrientation:
    def test_zero_rate_for_static_orientation(self):
        n = 4
        ts = np.linspace(0, 3, n)
        same_q = Quaternion.from_axis_angle(Point(0.3, -0.1, 0.2))
        r = Quaternion(np.tile(same_q.data, (n, 1)))
        q_rate = Point(np.zeros(n), np.zeros(n), np.zeros(n))

        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        query = np.linspace(ts[0], ts[-1], 10)
        out, rate = interp(query)

        np.testing.assert_allclose(rate.data, np.zeros_like(rate.data), atol=1e-6)


class TestSquadRateContinuity:
    def test_rate_does_not_spike_across_segment_boundary(self):
        """For constant angular rate the true rate is flat everywhere, so
        the central-differenced rate shouldn't show a large discontinuity
        right at a knot -- this would catch an eps/segment-lookup bug at
        the boundary that the orientation-only continuity test above
        wouldn't necessarily surface."""
        w = np.pi / 6
        r, q_rate, ts = constant_rate_z_rotation(5, w, 4.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        knot = ts[2]
        around = np.linspace(knot - 0.05, knot + 0.05, 11)
        out, rate = interp(around)

        # every sampled rate should stay close to the true constant rate,
        # with no jump as the query straddles the knot itself
        np.testing.assert_allclose(rate.z, np.full(len(around), w), atol=1e-3)
        np.testing.assert_allclose(rate.x, np.zeros(len(around)), atol=1e-3)
        np.testing.assert_allclose(rate.y, np.zeros(len(around)), atol=1e-3)


class TestSquadRateExactAtKnots:
    """Directly exercises the property squad_control_points is now
    designed to guarantee: the interpolated curve's rate at each knot
    matches the tangent (input q_rate) that was supplied for that knot,
    not just the position."""

    def test_rate_at_interior_knots_matches_supplied_tangent(self):
        w = np.pi / 5
        r, q_rate, ts = constant_rate_z_rotation(6, w, 5.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        # interior knots only -- ts[0] and ts[-1] are one-sided in the
        # central difference and covered separately below
        interior_ts = ts[1:-1]
        out, rate = interp(interior_ts)

        np.testing.assert_allclose(rate.z, np.full(len(interior_ts), w), atol=1e-3)
        np.testing.assert_allclose(rate.x, np.zeros(len(interior_ts)), atol=1e-3)
        np.testing.assert_allclose(rate.y, np.zeros(len(interior_ts)), atol=1e-3)

    def test_rate_at_domain_endpoints_matches_supplied_tangent(self):
        """ts[0] and ts[-1] are the cases where the central difference's
        eps window gets clamped to one side of the domain -- worth
        checking on its own since it's a different code path from the
        interior-knot case."""
        w = np.pi / 5
        r, q_rate, ts = constant_rate_z_rotation(6, w, 5.0)
        t = make_time(ts)
        interp = Quaternion.squad(r, q_rate, t)

        out, rate = interp(np.array([ts[0], ts[-1]]))

        np.testing.assert_allclose(rate.z, [w, w], atol=1e-3)
        np.testing.assert_allclose(rate.x, [0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(rate.y, [0.0, 0.0], atol=1e-3)