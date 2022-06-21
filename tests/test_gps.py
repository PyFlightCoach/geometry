from pytest import approx
from geometry.gps import GPS

import numpy as np

def test_offset():
    c = GPS( 52.542375, -1.631038)
    p = GPS( 52.542264, -1.631817)
    diff = c - p
    c2 = p.offset(diff)

    diff2 = c2 - p

    np.testing.assert_array_almost_equal(diff.data, diff2.data, 1e-4)

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

def test_diff():
    p0 = GPS( 50.201, 4.195)
    p0n = GPS( 50.206, 4.195) #directly north of p0
    
    diff= p0 - p0n # should be south vector
    assert diff.y == approx(0)
    assert diff.x < 0
    
    p0e= GPS( 50.201, 4.196)
    diff= p0 - p0e # should be west vector
    assert diff.x == approx(0)
    assert diff.y < 0




def test_sub():
    centre = GPS( 52.542375, -1.631038)
    pilot = GPS( 52.542264, -1.631817)

    vec = centre - pilot 

    assert abs(vec)[0] == approx(54.167, 1e-3)
    assert vec.x[0] == approx(12.3563, 1e-3)
    assert vec.y[0] == approx(52.7393, 1e-3)
