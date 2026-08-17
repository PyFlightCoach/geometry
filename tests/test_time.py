import numpy as np
from pytest import raises

from geometry import Time
from geometry.base import ExtrapolationError
from geometry.checks import assert_almost_equal


def test_time_interpolate_gets_the_right_dt():
    t = Time.from_t(np.arange(5) / 10)
    # i t   dt
    # 0 0.0 0.1
    # 1 0.1 0.1
    # 2 0.2 0.1
    # 3 0.3 0.1
    # 4 0.4 0.1

    assert_almost_equal(Time(0, 0.0), t.linterp_recaclulate_dt()(0))
    assert_almost_equal(Time(0.05, 0.0), t.linterp_recaclulate_dt()(0.5))
    assert_almost_equal(Time(0.4, 0.0), t.linterp_recaclulate_dt()(4))
    with raises(ExtrapolationError):
        t.linterp_recaclulate_dt()(5)


def test_time_getindex():
    t = Time.from_t(np.arange(5) / 10)
    assert t.get_index(0) == 0
    assert t.get_index(0.05) == 0.5
    np.testing.assert_array_equal(t.get_index([0.2, 0.25]), np.array([2, 2.5]))
    assert t.get_index(0.05, True) == 0
    assert t.get_index(0.06, True) == 1

    with raises(ValueError):
        t.get_index(-0.01, False, "throw")
    with raises(ValueError):
        t.get_index(np.array([-0.1, 0.3]), False, "throw")

    assert t.get_index(-0.01, False, "nearest") == 0
    np.testing.assert_array_equal(
        t.get_index(np.array([-0.1, 0.3]), False, "nearest"), np.array([0, 3])
    )

    assert np.isnan(t.get_index(-0.01, False, "nan"))
    assert np.isnan(t.get_index(np.array([-0.1, 0.3]), False, "nan")[0])


def test_time_get_value():
    t = Time.from_t(np.arange(5) / 10)
    assert t.get_value(0) == 0
    assert t.get_value(0.5) == 0.05
    np.testing.assert_array_equal(t.get_value(np.array([2, 2.5])), np.array(np.array([0.2, 0.25])))

    assert np.isnan(t.get_value(5))
    assert np.isnan(t.get_value(np.array([0, 5])))[1]