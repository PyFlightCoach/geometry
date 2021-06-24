import unittest
from geometry import Point, Points, Line
import numpy as np


class test_line(unittest.TestCase):
    def test_fit_points(self):
        ps = Points(np.random.random((1000, 3)))
        ps = ps + Points(
            np.array([
                np.linspace(0, 20, 1000),
                np.linspace(0, 20, 1000),
                np.linspace(0, 20, 1000),
            ]).T
        )
        line = Line.fit_points(ps)
        np.testing.assert_array_almost_equal(
            line.start.to_list(), 
            np.zeros(3), 
            0
        )
        np.testing.assert_array_almost_equal(
            line.end.to_list(), 
            np.full(3, 20.0), 
            0
        )







if __name__ == "__main__":
    unittest.main()