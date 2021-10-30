import unittest
from geometry import Coord
import numpy as np


class TestCoord(unittest.TestCase):
    def test_from_rotation_matrix(self):
        np.testing.assert_array_equal(
            Coord.from_nothing().rotation_matrix,
            np.identity(3)
        )

 