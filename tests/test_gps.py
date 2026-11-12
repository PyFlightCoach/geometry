import numpy as np
from geometry.gps import GPS
from pytest import approx


def test_offset():
    c = GPS(52.542375, -1.631038, 0)
    p = GPS(52.542264, -1.631817, 0)
    diff = c - p
    c2 = p.offset(diff)

    diff2 = c2 - p

    np.testing.assert_array_almost_equal(diff.data, diff2.data, 1e-4)


def test_diff_north_and_west():
    p0 = GPS(50.206, 4.1941755999999994, 0)
    p0n = GPS(50.201, 4.1941755999999994, 0)

    # Latitude increases northward, so subtracting the southern point gives
    # a northward (positive x) vector; equal longitude gives zero east offset.
    diff = p0 - p0n
    assert diff.y == approx(0)
    assert diff.x > 0

    p0e = GPS(50.206, 4.195, 0)
    # p0e has a larger (more easterly) longitude, so this difference points
    # west (negative y); equal latitude gives zero north offset.
    diff = p0 - p0e
    assert diff.x == approx(0)
    assert diff.y < 0


def test_diff_south_and_west():
    p0 = GPS(50.201, 4.195, 0)
    p0n = GPS(50.206, 4.195, 0)  # directly north of p0

    # p0n is north of p0, so p0 - p0n points south (negative x).
    diff = p0 - p0n
    assert diff.y == approx(0)
    assert diff.x < 0

    p0e = GPS(50.201, 4.196, 0)
    # p0e is east of p0, so p0 - p0e points west (negative y).
    diff = p0 - p0e
    assert diff.x == approx(0)
    assert diff.y < 0


def test_sub():
    centre = GPS(52.542375, -1.631038, 0)
    pilot = GPS(52.542264, -1.631817, 0)

    vec = centre - pilot

    assert abs(vec)[0] == approx(54.167, 1e-3)
    assert vec.x[0] == approx(12.3563, 1e-3)
    assert vec.y[0] == approx(52.7393, 1e-3)
