import unittest
from geometry.gps import GPSPosition
import numpy as np

class TestGPSPosition(unittest.TestCase):
    def test_offset(self):
        p = GPSPosition( 50.2066727, 4.1941755999999994)
        c = GPSPosition( 50.206507599999995,4.194035600484306)
        diff = c - p
        c2 = p.offset(diff)

        diff2 = c2 - p

        np.testing.assert_array_almost_equal(diff.to_list(), diff2.to_list())

#        self.assertAlmostEqual(off.lat, c.lat)
#        self.assertAlmostEqual(off.lon, c.lo)

    def test_diff(self):
        p0 = GPSPosition( 50.206, 4.1941755999999994)
        p0n = GPSPosition( 50.201, 4.1941755999999994)
        
        diff= p0 - p0n # should be south vector
        self.assertAlmostEqual(diff.y, 0)
        self.assertLess(diff.x, 0)
        
        p0e= GPSPosition( 50.206, 4.195)
        diff= p0 - p0e # should be west vector
        self.assertAlmostEqual(diff.x, 0)
        self.assertLess(diff.y, 0)


