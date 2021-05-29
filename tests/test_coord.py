import unittest
from geometry import Coord
import numpy as np


class TestCoord(unittest.TestCase):
    def test_from_rotation_matrix(self):
        np.testing.assert_array_equal(
            Coord.from_nothing().rotation_matrix,
            np.identity(3)
        )

    def test_inverse_rotation_matrix(self):
        coord = Coord.from_nothing().rotate(np.random.random((3,3)))
        np.testing.assert_array_equal(
            coord.inverse_rotation_matrix,
            np.linalg.inv(coord.rotation_matrix)
        )