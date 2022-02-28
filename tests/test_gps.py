import pytest
from geometry.gps import GPSPosition
from geometry.gps_positions import GPSPositions
import numpy as np

def test_offset():
    p = GPSPosition( 50.2066727, 4.1941755999999994)
    c = GPSPosition( 50.206507599999995,4.194035600484306)
    diff = c - p
    c2 = p.offset(diff)

    diff2 = c2 - p

    np.testing.assert_array_almost_equal(diff.to_list(), diff2.to_list())

def test_diff():
    p0 = GPSPosition( 50.206, 4.1941755999999994)
    p0n = GPSPosition( 50.201, 4.1941755999999994)
    
    diff= p0 - p0n # should be south vector
    pytest.approx(diff.y, 0)
    assert diff.x >  0
    
    p0e= GPSPosition( 50.206, 4.195)
    diff= p0 - p0e # should be west vector
    pytest.approx(diff.x, 0)
    assert diff.y < 0




def test_diff_plural():
    p0 = GPSPosition( 50.2066727, 4.1941755999999994)
    p0n = GPSPosition(50.206507599999995,4.194035600484306)

    p0s = GPSPositions.full(p0, 10)
    p0ns = GPSPositions.full(p0n, 10)

    diff= p0 - p0n
    diffs= p0s - p0ns

    np.testing.assert_array_equal(diff.to_list(), diffs[0].to_list())

def test_offset_plural():
    p = GPSPosition( 50.2066727, 4.1941755999999994)
    c = GPSPosition( 50.206507599999995,4.194035600484306)
    diff = c - p
    c2 = p.offset(diff)

    ps = GPSPositions.full(p, 10)
    cs = GPSPositions.full(c, 10)
    diffs = cs - ps
    c2s = ps.offset(diffs)

    np.testing.assert_array_almost_equal(c2.to_list(), c2s[0].to_list())

def test_longitude_scale():
    p = GPSPosition( 50.206507599999995,4.194035600484306)
    ps = GPSPositions.full(p, 10)
    assert p._longitude_scale == ps[0]._longitude_scale