from geometry.point import Point, cross, PX, PY, PZ, P0, is_perpendicular

import unittest
from math import pi
from pytest import mark, approx, fixture, raises
import numpy as np
from geometry.testing import assert_equal

def test_init():
    p = Point(1,2,3)
    assert p.x == 1
    assert p.y == 2

def test_cross():
    a = Point(1,2,3)
    b = Point(1,2,3)

    c = a.cross(b)

    np.testing.assert_almost_equal(c.data[0,:], np.cross(a.data[0,:], b.data[0,:]))
    assert a.cross(b) == cross(a, b)

def test_cross2():
    a = Point(np.random.random((10, 3)))
    b = Point(np.random.random((10, 3)))

    c = a.cross(b)

    c_check = np.array([np.cross(_a.data[0], _b.data[0]) for _a, _b in zip(a,b)])

    np.testing.assert_array_almost_equal(c.data, c_check)


def test_scale():
    assert Point(1,0,0).scale(5) == Point(5,0,0)
    assert Point(1,0,0).scale(5).data.shape == Point(5,0,0).data.shape

    with np.errstate(divide="raise"):
        assert Point.concatenate([PX(1,10), P0(10)]).scale(5) == Point.concatenate([PX(5,10), P0(10)])


def test_unit():
    assert Point(5,0,0).unit() == Point(1,0,0)
    assert Point(5,0,0).unit().data.shape == (1,3)

def test_angle_between():
    #with raises(ValueError):
    Point(1,2,3).cos_angle_between(Point.zeros()) == P0()

    assert PX().cos_angle_between(PY()) == 0

    assert PX().angle_between(PY()) == np.pi / 2

    
    assert PX(-1).angle_between(Point(1,1,0)) == 3 * np.pi / 4


def test_is_parallel():
    assert not Point(1,0, 0).is_parallel(Point(-1,0, 0))
    assert Point(1,0, 0).is_parallel(Point(1,0, 0))
    assert not Point(1,0, 0).is_parallel(Point(0,1, 0))


def test_scalar_projection():
    assert Point(1, 1, 0).scalar_projection(PX()) == 1
    assert P0().scalar_projection(Point(1, 1, 0)) == 0
    assert Point(1, 1, 0).scalar_projection(P0()) == 0

def test_is_perpendicular():
    assert is_perpendicular(Point(1, 0, 0), Point(0, 1, 0))
    assert is_perpendicular(Point(1, 0, 0), Point(0, 0, 1))
    assert is_perpendicular(Point(0, 1, 0), Point(1, 0, 0))
    assert not is_perpendicular(Point(1, 0, 0), Point(1, 1, 0))


def test_full():
    points = Point.full(Point(1,2,3), 100)
    assert len(points) == 100
    assert isinstance(points, Point)
    assert points[20] == Point(1, 2, 3)

    
def test_X():
    assert Point.X(1,100) + Point.Y(1,100) + Point.Z(1,100) == Point(np.ones((100,3)))


def test_to_rotation_matrix():
    np.testing.assert_array_equal(P0().to_rotation_matrix()[0],np.identity(3))


def test_vector_projection():
    res = Point.vector_projection(PX(1,20), PY(1))
    assert res == P0()

    res = Point.vector_projection(PZ(20), PZ(20,20))
    assert res == PZ(20)


def test_vector_rejection():
    assert Point.vector_rejection(PX(), PX()) == P0()
    assert Point.vector_rejection(PY(), PX()) == PY()
    assert Point.vector_rejection(Point(1,1,0), PX()) == PY()
    assert Point.vector_rejection(Point(1,1,1), PX()) == Point(0, 1, 1)


def test_remove_outliers():
    ps = P0(100)

    ps.data[50,:] = [1000,1000,1000]

    sps = ps.remove_outliers()

    assert_equal(sps, P0(100))


def test_remove_outliers_fail():
    import pandas as pd
    p = Point(pd.read_csv('tests/test_remove_outliers.csv').iloc[:,1:])
    assert np.sum(np.isnan(p.data)) == 0