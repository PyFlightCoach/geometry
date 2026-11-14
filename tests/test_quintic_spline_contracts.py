"""Quintic Hermite interpolation against an independent global polynomial."""

import numpy as np
from numpy.polynomial import Polynomial
import pytest

from geometry import Point, QuinticHermiteSpline


POLYNOMIALS = (
    Polynomial([1, 2, -3, .5, .25, -.125]),
    Polynomial([-2, 0, 1, 0, -.25, 0]),
    Polynomial([3, -2, 0, 1, 0, -.2]),
)


def _analytic(times, order=0):
    return np.column_stack([poly.deriv(order)(np.atleast_1d(times)) for poly in POLYNOMIALS])


@pytest.fixture
def spline():
    times = np.array([-.5, .25, 1.5, 3.])
    return QuinticHermiteSpline(times, *[Point(_analytic(times, order)) for order in range(3)])


def test_nonuniform_segments_reproduce_all_three_polynomials_and_derivatives(spline):
    # A quintic with its endpoint p, p' and p'' constraints is unique. The
    # independent power-basis polynomials therefore determine every interior
    # value and derivative, including the non-unit segment time scaling.
    times = np.array([-.5, -.4, 0., .25, .3, 1.1, 1.5, 2.4, 3.])
    before = [value.copy() for value in (spline.time, spline.pos.data, spline.vel.data, spline.acc.data)]
    for order, actual in enumerate(spline(times)):
        np.testing.assert_allclose(actual.data, _analytic(times, order), rtol=0, atol=2e-12)
    for value, original in zip((spline.time, spline.pos.data, spline.vel.data, spline.acc.data), before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("order,method", [(0, "pos_spline"), (1, "vel_spline"), (2, "acc_spline")])
def test_individual_accessors_evaluate_values_endpoints_and_clipped_queries(spline, order, method):
    query = np.array([-100., -.5, .25, .6, 1.5, 3., 100.])
    expected = _analytic(np.clip(query, -.5, 3.), order)
    np.testing.assert_allclose(getattr(spline, method)(query).data, expected, rtol=0, atol=2e-12)


def test_scalar_and_empty_queries_keep_point_shapes(spline):
    for order, value in enumerate(spline(.6)):
        assert value.data.shape == (1, 3)
        np.testing.assert_allclose(value.data, _analytic(.6, order), rtol=0, atol=2e-12)
    for value in spline(np.array([])):
        assert value.data.shape == (0, 3)


@pytest.mark.parametrize("bad", ["position", "velocity", "acceleration"])
def test_constraint_lengths_must_match_the_knots(bad):
    times = np.array([0., 1., 2.])
    constraints = [Point(_analytic(times, order)) for order in range(3)]
    constraints[["position", "velocity", "acceleration"].index(bad)] = Point([[0., 0., 0.]])
    with pytest.raises(AssertionError):
        QuinticHermiteSpline(times, *constraints)
