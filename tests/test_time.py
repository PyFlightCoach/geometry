from geometry import Time
from pytest import fixture, raises
import numpy as np


def test_time_interpolate():
    t = Time.from_t(np.arange(5))

    t1 = t.interpolate(t.t)([2.5])

    assert t1.t[0] == 2.5
    assert t1.dt[0] == 0.5


def test_time_interpolate_t():
    t = Time.from_t(np.arange(5)/2)

    assert 1 == t.interpolate_t(0.5)
    assert 1.5 == t.interpolate_t(0.75)
    with raises(ValueError):
        t.interpolate_t(5)
    with raises(ValueError):
        t.interpolate_t(-1)
    
    