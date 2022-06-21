import unittest

from geometry.points import Points

import numpy as np
import pandas as pd

from geometry import Point


class TestPoints(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]])
        self.pnts = Points(self.points)

    def test_xyz(self):
        self.assertCountEqual(
            self.pnts.x,
            np.vectorize(lambda *args: Point(*args).x)(*self.points.T)
        )

        self.assertCountEqual(
            self.pnts.y,
            np.vectorize(lambda *args: Point(*args).y)(*self.points.T)
        )
        self.assertCountEqual(
            self.pnts.z,
            np.vectorize(lambda *args: Point(*args).z)(*self.points.T)
        )

    def test_count(self):
        self.assertEqual(self.pnts.count, 4)

    def test_abs(self):
        self.assertCountEqual(
            abs(self.pnts),
            np.vectorize(lambda *args: abs(Point(*args)))(*self.points.T)
        )

    def test_add(self):

        pnts = Points(np.random.random((100, 3)))

        np.testing.assert_array_equal(
            (pnts + pnts).data,
            np.array(np.vectorize(
                lambda *args: tuple(Point(*args) + Point(*args))
            )(*pnts.data.T)).T
        )

        np.testing.assert_array_equal(
            (pnts + Point(1, 1, 1)).data,
            np.array(np.vectorize(
                lambda *args: tuple(Point(*args) + Point(1, 1, 1))
            )(*pnts.data.T)).T
        )

        np.testing.assert_array_equal(
            (Point(1, 1, 1) + pnts).data,
            np.array(np.vectorize(
                lambda *args: tuple(Point(1, 1, 1) + Point(*args))
            )(*pnts.data.T)).T
        )

    def test_mul_points_points(self):
        def sqr(*args):
            return Point(*args) * Point(*args)

        self.assertCountEqual(
            (self.pnts * self.pnts).x,
            np.vectorize(lambda *args: sqr(*args).x)(*self.points.T)
        )
        self.assertCountEqual(
            (self.pnts * self.pnts).y,
            np.vectorize(lambda *args: sqr(*args).y)(*self.points.T)
        )

    def test_mul_points_scalar(self):
        def dbl(*args):
            return 2 * Point(*args)

        self.assertCountEqual(
            (2 * self.pnts).x,
            np.vectorize(lambda *args: dbl(*args).x)(*self.points.T)
        )

    def test_mul_points_array(self):
        pnts = Points(np.random.random((100, 3)))
        mults = np.random.random(100)

        def mul(*args):
            return tuple(Point(*args[:3]) * args[3])

        np.testing.assert_array_equal(
            (mults * pnts).data,
            np.array(np.vectorize(
                lambda *args: mul(*args))(*np.column_stack([pnts.data, mults]).T
                                          )).T
        )
        with self.assertRaises(NotImplementedError):
            np.testing.assert_equal(
                self.pnts * np.random.random(5),
                NotImplemented
            )

    def test_neg(self):
        self.assertCountEqual(
            (-self.pnts).y,
            np.vectorize(lambda *args: -Point(*args).y)(*self.points.T)
        )

    def test_truediv_scalar(self):
        self.assertCountEqual(
            (self.pnts / 2).x,
            np.vectorize(lambda *args: (Point(*args) / 2).x)(*self.points.T)
        )

    def test_truediv_array(self):
        pnts = Points(np.random.random((100, 3)))
        denoms = np.random.random(100)
        np.testing.assert_array_equal(
            (pnts / denoms).data,
            np.array(np.vectorize(
                lambda *args: tuple(Point(*args[0:3]) / args[3])
            )(*np.column_stack([pnts.data, denoms]).T)).T
        )

    def test_truediv_points_points(self):
        pass

    def test_scale_scalar(self):
        self.assertCountEqual(
            abs(self.pnts.scale(1)),
            np.vectorize(lambda *args: abs(Point(*args).scale(1))
                         )(*self.points.T)
        )

    def test_trigs(self):
        np.testing.assert_array_equal(
            self.pnts.sines().data,
            np.array(np.vectorize(
                lambda *args: tuple(Point(*args).sines)
            )(*self.points.T)).T
        )

        np.testing.assert_array_equal(
            self.pnts.cosines().data,
            np.array(np.vectorize(
                lambda *args: tuple(Point(*args).cosines)
            )(*self.points.T)).T
        )

    def test_dot(self):
        self.assertCountEqual(
            self.pnts.dot(self.pnts),
            np.vectorize(lambda *args: dot_product(Point(*args),
                                                   Point(*args)))(*self.points.T)
        )

    def test_cross(self):
        np.testing.assert_array_equal(
            self.pnts.cross(self.pnts).data,
            np.array(np.vectorize(
                lambda *args: tuple(cross_product(Point(*args), Point(*args)))
            )(*self.points.T)).T
        )

    def test_from_pandas(self):
        arr = np.random.random((100, 3))
        df = pd.DataFrame(arr, columns=list('xyz'))
        pnts = Points.from_pandas(df)
        np.testing.assert_array_equal(arr, pnts.data)

    def test_to_pandas(self):
        pnts = Points(np.random.random((100, 3)))
        df = pnts.to_pandas('a', 'b')
        np.testing.assert_array_equal(df.columns, ['axb', 'ayb', 'azb'])

    def test_diff(self):
        rawd, step0 = np.linspace(0, 100, 100, retstep=True)
        pins = Points(np.array([rawd, rawd * 2, rawd * -1]).T)
        dt=0.5
        vels = pins.diff(np.full(100, dt))
        np.testing.assert_array_almost_equal(
            vels.data, 
            np.tile([step0 / dt, 2 * step0 / dt, -1 * step0 / dt], (100, 1))
        )

    def test_getitem(self):
        np.testing.assert_array_equal(self.pnts[0].to_list(),[1, 0, 0])
        np.testing.assert_array_equal(self.pnts[-1].to_list(),[1, 2, 3])
        


    def test_iter(self):
        for point in self.pnts:
            self.assertIsInstance(point, Point)

    def test_mean(self):
        mean = self.pnts.mean()

        self.assertEqual(mean.x, np.mean(self.pnts.x))


    def test_minloc(self):
        points = Points.full(Point(1,1,1), 100) * np.concatenate([np.linspace(50,0, 50), np.linspace(0,50, 50)], axis=0)
        minlocs =  points.minloc()
        #assert isinstance(minlocs.x, int)
        np.testing.assert_array_equal(minlocs.to_list(), [49,49,49])

    def test_slice(self):
        points = Points.full(Point(1,1,1), 100) * np.linspace(0,99, 100)
        np.testing.assert_array_equal(points[:50].data, (Points.full(Point(1,1,1), 50) * np.linspace(0,49, 50)).data)

if __name__ == "__main__":
    unittest.main()