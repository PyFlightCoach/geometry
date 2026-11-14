"""Nonuniform sample ownership, endpoint/extrapolation and rescaling contracts."""

import importlib

import numpy as np
import pytest

from geometry import Time
from geometry.base import ExtrapolationError


def test_nonuniform_forward_intervals_end_at_zero_without_mutating_input():
    source = np.array([4., 4.25, 6.])
    time = Time.from_t(source)
    np.testing.assert_array_equal(time.data, [[4., .25], [4.25, 1.75], [6., 0.]])
    np.testing.assert_array_equal(source, [4., 4.25, 6.])
    assert time.duration == 2


def test_scaling_reset_and_translation_preserve_their_distinct_time_meanings():
    time = Time.from_t([4., 4.25, 6.])
    before = time.data.copy()
    np.testing.assert_array_equal(time.scale(6).data, [[4, .75], [4.75, 5.25], [10, 0]])
    np.testing.assert_array_equal(time.reset_zero().data, [[0, .25], [.25, 1.75], [2, 0]])
    np.testing.assert_array_equal((time + 2).data, [[6, .25], [6.25, 1.75], [8, 0]])
    np.testing.assert_array_equal((time - 2).data, [[2, .25], [2.25, 1.75], [4, 0]])
    np.testing.assert_array_equal(time.data, before)


def test_extension_uses_last_real_interval_or_the_explicit_interval():
    time = Time.from_t([4., 4.25, 6.])
    np.testing.assert_array_equal(time.extend().data, [[4, .25], [4.25, 1.75], [6, 1.75], [7.75, 0]])
    np.testing.assert_array_equal(time.extend(.5).data, [[4, .25], [4.25, 1.75], [6, .5], [6.5, 0]])
    singleton = Time.from_t(5.)
    with pytest.raises(ValueError, match="only one point"):
        singleton.extend()
    np.testing.assert_allclose(singleton.extend(.4).data, [[5., .4], [5.4, 0]], rtol=0, atol=1e-15)


@pytest.mark.parametrize("npoints,minpoints,freq,expected", [
    (None, 1, 4, [0., 1 / 3, 2 / 3, 1.]),
    (3, 1, 100, [0., .5, 1.]),
    (None, 3, .1, [0., .5, 1.]),
])
def test_uniform_time_has_requested_count_and_closed_endpoints(npoints, minpoints, freq, expected):
    time = Time.uniform(1., npoints, minpoints=minpoints, freq=freq)
    np.testing.assert_array_equal(time.t, expected)
    np.testing.assert_allclose(time.dt, np.r_[np.diff(expected), 0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("snap", [False, True])
def test_nonuniform_indices_handle_both_outside_endpoints_under_each_policy(snap):
    time = Time.from_t([10., 11., 13., 20.])
    query = np.array([8., 10., 10.5, 12., 16.5, 20., 21.])
    inside = [0, 0, 1, 2, 3] if snap else [0, .5, 1.5, 2.5, 3]
    expected_nan = np.array([np.nan, *inside, np.nan])
    np.testing.assert_allclose(time.get_index(query, snap, "nan"), expected_nan, rtol=0, atol=1e-15, equal_nan=True)
    np.testing.assert_array_equal(time.get_index(query, snap, "nearest"), [0, *inside, 3])
    with pytest.raises(ValueError, match="out of bounds"):
        time.get_index(query, snap, "throw")
    for value, expected in ((8., 0), (21., 3)):
        assert time.get_index(value, snap, "nearest") == expected
        assert np.isnan(time.get_index(value, snap, "nan"))
        with pytest.raises(ValueError, match="out of bounds"):
            time.get_index(value, snap, "throw")


def test_negative_and_fractional_indices_refer_to_exact_nonuniform_samples():
    time = Time.from_t([10., 11., 13., 20.])
    query = np.array([-4., -3., -2.5, -1., 0., 1.5, 3., 4.])
    np.testing.assert_allclose(time.get_value(query), [10, 11, 12, 20, 10, 12, 20, np.nan], rtol=0, atol=1e-15, equal_nan=True)


def test_interpolation_recalculates_intervals_from_requested_times():
    time = Time.from_t([0., 2., 5.])
    interpolate = time.linterp_recaclulate_dt()
    np.testing.assert_array_equal(interpolate(np.array([0., .5, 1., 1.5, 2.])).data,
                                  [[0, 1], [1, 1], [2, 1.5], [3.5, 1.5], [5, 0]])
    with pytest.raises(ExtrapolationError):
        interpolate(np.array([-1., 2.]))
    nearest = time.linterp_recaclulate_dt(extrapolate="nearest")
    np.testing.assert_array_equal(nearest(np.array([-1., 3.])).data, [[0, 5], [5, 0]])


def test_concatenation_recalculates_the_gap_between_nonuniform_inputs():
    result = Time.concatenate([Time.from_t([0., .5]), Time.from_t([2., 3.])])
    np.testing.assert_array_equal(result.data, [[0, .5], [.5, 1.5], [2, 1], [3, 0]])


def test_now_uses_the_clock_as_a_singleton_timestamp(monkeypatch):
    monkeypatch.setattr(importlib.import_module("geometry.time"), "time", lambda: 1234.5)
    np.testing.assert_array_equal(Time.now().data, [[1234.5, 0]])
