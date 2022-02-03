from geometry import Point
from geometry.point import  angle_between, is_parallel, scalar_projection, is_perpendicular, min_angle_between, dot_product
import unittest
from math import pi


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.p0 = Point(x=0, y=0, z=0)
        self.px = Point(x=1, y=0, z=0)
        self.pxy = Point(x=1, y=1, z=0)
        self.px_minus = Point(x=-3, y=0, z=0)
        self.py = Point(x=0, y=100, z=0)
        self.pz = Point(x=0, y=0, z=11)
        self.palmostequal = Point(x=1.1, y=1.05, z=1.15)
        self.pdiv1 = Point(x=1, y=2, z=3)
        self.pdiv2 = Point(x=1, y=4, z=9)

    def test_angle_betweenzero_length(self):
        with self.assertRaises(ValueError):
            angle_between(self.p0, self.p0)

    def test_angle_between(self):
        self.assertAlmostEqual(angle_between(self.px, self.pxy), pi / 4)

    def test_angle_between_90(self):
        self.assertAlmostEqual(angle_between(self.px, self.py), pi / 2)

    def test_angle_between_opposite(self):
        self.assertAlmostEqual(angle_between(self.px, self.px_minus), pi)

    def test_angle_between_more_than_90(self):
        self.assertAlmostEqual(angle_between(
            self.px_minus, self.pxy), 3 * pi / 4)

    def test_min_angle_between_more_than_90(self):
        self.assertAlmostEqual(min_angle_between(
            self.px_minus, self.pxy), pi / 4)

    def test_is_equal(self):
        self.assertTrue(self.p0.is_equal())
        self.assertTrue(self.palmostequal.is_equal(0.1))
        self.assertFalse(self.palmostequal.is_equal(0.025))

    def test_is_parallel(self):
        self.assertFalse(is_parallel(
            Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
            Point(x=0, y=-0.5150380749100639, z=0.8571673007021064)
        ))

        with self.assertRaises(TypeError):
            is_parallel(Point(0, 0, 0), Point(0, 0,))

        self.assertTrue(
            is_parallel(
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                0.001
            )
        )
        self.assertTrue(
            is_parallel(
                Point(x=0, y=0.5, z=0.5),
                Point(x=0, y=-1, z=-1)
            )
        )

    def test_is_anti_parallel(self):
        self.assertFalse(
            is_parallel(
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                Point(x=0, y=-0.5150380749100639, z=0.8571673007021064)
            )
        )
        self.assertTrue(
            is_parallel(
                Point(x=1, y=1, z=1),
                Point(x=1, y=1, z=1)
            )
        )
        self.assertTrue(
            is_parallel(
                Point(x=0, y=0.8571673007021064, z=0.5150380749100639),
                Point(x=0, y=-0.8571673007021064, z=-0.5150380749100639),
                0.001
            )
        )

    def test_eq(self):
        self.assertTrue(Point(0, 0, 0) == Point(0, 0, 0))
        self.assertTrue(Point(1, 2, 3) == Point(1, 2, 3))

    def test_abs(self):
        self.assertEqual(abs(Point(0, 0, 0)), 0)
        self.assertEqual(abs(Point(1, 0, 0)), 1)
        self.assertEqual(abs(Point(0, 100, 0)), 100)

    def test_div(self):
        self.assertAlmostEqual(self.pdiv2 / self.pdiv1, Point(x=1, y=2, z=3))
        self.assertAlmostEqual(self.pdiv2 / 1, self.pdiv2)

    def test_rotate(self):
        self.assertEqual(
            self.px.rotate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            self.px
        )
        self.assertEqual(
            self.px.rotate([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            Point(0, 1, 0)
        )

    def test_scalar_projection(self):
        self.assertEqual(scalar_projection(Point(1, 1, 0), Point(1, 0, 0)), 1)
        self.assertEqual(scalar_projection(Point(0, 0, 0), Point(1, 1, 0)), 0)
        self.assertEqual(scalar_projection(Point(1, 1, 0), Point(0, 0, 0)), 0)

    def test_is_perpendicular(self):
        self.assertTrue(is_perpendicular(Point(1, 0, 0), Point(0, 1, 0)))
        self.assertTrue(is_perpendicular(Point(1, 0, 0), Point(0, 0, 1)))
        self.assertTrue(is_perpendicular(Point(0, 1, 0), Point(1, 0, 0)))
        self.assertFalse(is_perpendicular(Point(1, 0, 0), Point(1, 1, 0)))

    def test_dot_product(self):
        self.assertEqual(dot_product(Point(1,1,1), Point(1,1,1)), 3)
