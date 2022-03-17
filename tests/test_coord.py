import unittest
from geometry import Coord, Point, P0, PX, PY, PZ
import numpy as np



def test_axes():
    coord = Coord(np.ones((20, 12)))
    assert coord.origin == Point(np.ones((20,3)))

def test_from_axes():
    coord = Coord.from_axes(P0(2), PX(1,2), PY(1,2), PZ(1,2))
    assert coord.data[:,:3] == P0(2)

class TestCoord(unittest.TestCase):
    def test_from_rotation_matrix(self):
        np.testing.assert_array_equal(
            Coord.from_nothing().rotation_matrix(),
            np.identity(3).reshape((1,3,3))
        )

 