from pytest import approx
from geometry.gps import GPS

import numpy as np

def test_offset():
    p = GPS( 50.2066727, 4.1941755999999994)
    c = GPS( 50.206507599999995,4.194035600484306)
    diff = c - p
    c2 = p.offset(diff)

    diff2 = c2 - p

    np.testing.assert_array_almost_equal(diff.data, diff2.data)

#        self.assertAlmostEqual(off.lat, c.lat)
#        self.assertAlmostEqual(off.lon, c.lo)

def test_diff():
    p0 = GPS( 50.206, 4.1941755999999994)
    p0n = GPS( 50.201, 4.1941755999999994)
    
    diff= p0 - p0n # should be south vector
    assert diff.y == approx(0)
    assert diff.x < 0
    
    p0e= GPS( 50.206, 4.195)
    diff= p0 - p0e # should be west vector
    assert diff.x == approx(0)
    assert diff.y < 0


