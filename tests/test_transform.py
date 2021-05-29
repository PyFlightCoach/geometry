import unittest
from geometry import Coord, Transformation, Point, Quaternion, Points, Quaternions
import numpy as np


class TestTransformation(unittest.TestCase):
 

    def test_from_coords(self):
        c1 = Coord.from_xy(Point(1,0,0), Point(1,0,0), Point(0, 1, 0))
        c2 = Coord.from_xy(Point(1,0,0), Point(0,1,0), Point(1, 0, 0))
        trans_to = Transformation.from_coords(c1,c2)
        trans_from = Transformation.from_coords(c2,c1)
        
        ps = Points(np.random.random((100, 3)))
        
        np.testing.assert_array_almost_equal(
            ps.data,
            trans_from.translate(trans_to.translate(ps)).data
        )

        qs = Quaternions.from_euler(ps)
        np.testing.assert_array_almost_equal(
            qs.data,
            trans_from.quat(trans_to.quat(qs)).data
        )



        

    def test_translate(self):
        ca = Coord.from_nothing()
        cb = Coord.from_nothing().translate(Point(1, 0, 0))
        transform = Transformation.from_coords(ca, cb)
        self.assertEqual(transform.translate(Point(0, 0, 0)), Point(1, 0, 0))

    def _test_rotate(self, c1, c2, p1, p2):
        transform = Transformation.from_coords(c1, c2)
        p1b = transform.rotate(p1)
        self.assertEqual(p1b, p2, str(p1b) + ' != ' + str(p2))

    def test_rotate(self):
        self._test_rotate(
            Coord.from_nothing(),
            Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)),
            Point(1, 1, 0),
            Point(1, -1, 0)
        )

        self._test_rotate(
            Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)),
            Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)),
            Point(1, 1, 0),
            Point(-1, -1, 0)
        )



    def test_points(self):
        points = Points(np.random.random((100, 3)))
        transform = Transformation(
            Point(*np.random.random(3)),
            Quaternion(*np.random.random(4)).norm()
        )

        np.testing.assert_array_almost_equal(
            transform.rotate(points).data,
            np.array(np.vectorize(
                lambda *args: tuple(transform.rotation.transform_point(Point(*args)))
            )(*points.data.T)).T
        )

if __name__ == '__main__': 
    unittest.main()