import unittest
from pytest import mark, approx, raises
from geometry.quaternion import Quaternion
from geometry.point import Point

import numpy as np
from scipy.spatial.transform import Rotation as R



def test_mul():
    q1 = Quaternion(np.random.random((100, 4))).norm()
    q2 = Quaternion(np.random.random((100, 4))).norm()

    # Quaternion * Quaternion
    np.testing.assert_array_almost_equal(
        (q1 * q2).xyzw,
        np.array([(R.from_quat(_1.xyzw[0]) * R.from_quat(_2.xyzw[0])).as_quat() for _1, _2 in zip(q1, q2)]),
        err_msg="failed to do Quaternions * Quaternions"
    )

def test_from_euler():
    parr = np.random.random((20, 3))

    np.testing.assert_array_almost_equal(
        Quaternion.from_euler(Point(parr)).xyzw,
        R.from_euler('xyz', parr).as_quat()
    )



def test_to_euler():
    qarr = Quaternion(np.random.random((500, 4))).norm()

    earr = [R.from_quat(q.xyzw[0]).as_euler("xyz") for q in qarr]
    np.testing.assert_array_almost_equal(
        qarr.to_euler().data,
        earr
    )

@mark.skip("not sure what this is in scipy")
def test_norm():
    qarr = Quaternion(np.random.random((500, 4)))
    earr = [R.from_quat(q.xyzw[0]).as_quat() for q in qarr]
    np.testing.assert_array_almost_equal(
        qarr.norm().xyzw,
        earr
    )
@mark.skip("not sure if this exists either")
def test_conjugate():

    qarr = Quaternion(np.random.random((500, 4)))
    earr = [R.from_quat(q.xyzw[0]).conjugate() for q in qarr]
    np.testing.assert_array_almost_equal(
        qarr.conjugate().xyzw,
        earr
    )

def test_inverse():
    qarr = Quaternion(np.random.random((500, 4)))
    earr = [R.from_quat(q.xyzw[0]).inv().as_quat() for q in qarr]
    np.testing.assert_array_almost_equal(
        -qarr.inverse().xyzw,  # TODO not sure why the - is needed
        earr
    )

def test_body_diff():
    qs = Quaternion.from_euler(Point.zeros(100))
    qs = qs.body_rotate(Point.X(1, 100) * np.linspace(0,np.pi, 100))
    dt = np.ones(100)
    dq = qs.body_diff(dt)
    np.testing.assert_array_almost_equal(
        dq.data,
        Point.X(np.pi/100, 100).data
    )


def test_rotate():
    q = Quaternion.from_euler(Point(0, 0, 0))
    qdot = q.rotate(Point(0, 0, np.radians(5)))
    assert qdot.transform_point(Point(1, 0, 0)).y[0] == approx(np.sin(np.radians(5)))

def test_body_rotate():
    q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
    qdot = q.body_rotate(Point(np.radians(5), 0, 0))

    assert qdot.transform_point(Point(0, 1, 0)).z == np.sin(np.radians(5))

def test_body_rotate_zero():
    qinit = Quaternion.from_euler(Point(0, 0, 0))
    qdot = qinit.body_rotate(Point(0, 0, 0))

    np.testing.assert_array_equal(list(qinit), list(qdot))


def test_to_from_axis_angle():
    points = Point(np.random.random((100, 3)))
    tqs = Quaternion.from_axis_angle(points)
    tps = tqs.to_axis_angle()

    np.testing.assert_array_almost_equal(
        points.data,
        tps.data
    )

def test_to_axis_angle():
    q1 = Quaternion.from_euler(Point(0,0,np.pi/4))
    assert q1.to_axis_angle() == Point(0, 0, np.pi/4)



def test_axis_rates():
    q    = Quaternion.from_euler(Point(0.0, 0.0, np.pi/2))
    qdot = Quaternion.from_euler(Point(np.radians(5), 0.0, np.pi/2))

    rates = Quaternion.axis_rates(q, qdot)

    np.testing.assert_almost_equal(np.degrees(rates.data), Point(0,5,0).data)

def test_body_axis_rates():
    q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
    qdot = Quaternion.from_euler(Point(np.radians(5), 0, np.pi / 2))

    rates = Quaternion.body_axis_rates(q, qdot)
    np.testing.assert_almost_equal(np.degrees(rates.data), Point(5,0,0).data)



@mark.skip("to be thought about later")        
def test_from_rotation_matrix():

    rmats = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array(Point(1, 1, 0).to_rotation_matrix()),
        np.array(Point(0.7, -1.2, 1).to_rotation_matrix())
    ]

    for rmat in rmats:
        quat = Quaternion.from_rotation_matrix(rmat)

        rmat2 = quat.to_rotation_matrix()

        
        np.testing.assert_array_equal(rmat, rmat2)



