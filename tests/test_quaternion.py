import unittest
from geometry import Quaternion, Point
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation as R



testvals = np.array([0, 0, 0])



class TestQuaternion(unittest.TestCase):
  
    def test_from_euler(self):
        parr = np.random.random((20, 3))
        
        def test_func(row):
            return Quaternion.from_euler(Point(*row)).xyzw

        res = np.apply_along_axis(test_func,axis=1, arr=parr)

        # 'xyz' specifies xyz-fixed Euler angles, as used in ArduPilot
        spy = R.from_euler('xyz', parr).as_quat()
        spout = spy.copy()

        np.testing.assert_array_almost_equal(
            res,
            spout
        )

    def test_to_euler(self):
        qarr = np.random.random((500, 4))
        for i in range(0,len(qarr)):
            q = Quaternion(*qarr[i,:]).norm()
            sR = R.from_quat(q.xyzw)
            np.testing.assert_array_almost_equal(
                np.array(tuple(q.to_euler())),
                sR.as_euler("xyz")
            )
            

    def test_from_rotation_matrix(self):

        rmats = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            Point(1, 1, 0).to_rotation_matrix(),
            Point(0.7, -1.2, 1).to_rotation_matrix()
        ]

        for rmat in rmats:
            quat = Quaternion.from_rotation_matrix(np.array(rmat))

            rmat2 = quat.to_rotation_matrix()

            for i in range(0, 3):
                for j in range(0, 3):
                    self.assertAlmostEqual(rmat[i][j], rmat2[i][j])

    def test_iter(self):
        quat = Quaternion(1, 2, 3, 4)
        self.assertEqual(tuple(quat), (1, 2, 3, 4))
        self.assertEqual(list(quat), [1, 2, 3, 4])

    def test_to_axis_angle(self):
        q1 = Quaternion.from_euler(Point(0,0,np.pi/4))
        np.testing.assert_array_almost_equal(q1.to_axis_angle().to_list(), Point(0, 0, np.pi/4).to_list())


    def test_axis_rates(self):
        q    = Quaternion.from_euler(Point(0.0, 0.0, np.pi/2))
        qdot = Quaternion.from_euler(Point(np.radians(5), 0.0, np.pi/2))

        rates = Quaternion.axis_rates(q, qdot)

        np.testing.assert_array_almost_equal(
            (rates * 180 / np.pi).to_list(), [0.,5.,0.]
        )

    def test_body_axis_rates(self):
        q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
        qdot = Quaternion.from_euler(Point(np.radians(5), 0, np.pi / 2))

        rates = Quaternion.body_axis_rates(q, qdot)
        self.assertAlmostEqual(np.degrees(rates.x), 5)
        self.assertAlmostEqual(np.degrees(rates.y), 0)
        self.assertAlmostEqual(np.degrees(rates.z), 0)

    def test_eaxisrates(self):
        r = np.array(Point(np.pi / 2, 0, 0).to_rotation_matrix())
        rdot = np.array(Point(np.pi / 2 + np.radians(5),
                              0, 0).to_rotation_matrix())

        a = np.dot(rdot, r.T)
        self.assertAlmostEqual(a[0, 1], 0)
        self.assertAlmostEqual(a[0, 2], 0)
        self.assertAlmostEqual(np.degrees(a[1, 2]), -5, 1)

    def test_rotate(self):
        q = Quaternion.from_euler(Point(0, 0, 0))
        qdot = q.rotate(Point(0, 0, np.radians(5)))
        self.assertAlmostEqual(qdot.transform_point(
            Point(1, 0, 0)).y, np.sin(np.radians(5)))

    def test_body_rotate(self):
        q = Quaternion.from_euler(Point(0, 0, np.pi / 2))
        qdot = q.body_rotate(Point(np.radians(5), 0, 0))

        self.assertAlmostEqual(qdot.transform_point(
            Point(0, 1, 0)).z, np.sin(np.radians(5)))

    def test_body_rotate_zero(self):
        qinit = Quaternion.from_euler(Point(0, 0, 0))
        qdot = qinit.body_rotate(Point(0, 0, 0))

        np.testing.assert_array_equal(list(qinit), list(qdot))

    def test_from_axis_angle_zero(self):
        qinit = Quaternion.from_axis_angle(Point(0, 0, 0))
        np.testing.assert_array_equal(list(qinit), [1, 0, 0, 0])

        qb = Quaternion.from_axis_angle(Point(1, 0, 0))

        self.assertNotEqual(abs(qb), 0)

        pnts = np.concatenate(
            [np.zeros((1, 3)), np.random.random((5, 3))])

        #pnts = np.random.random((5, 3))

        def tfunc(*args):
            quat = Quaternion.from_axis_angle(Point(*args))
            return tuple(quat)

        quats = np.array(np.vectorize(tfunc)(*pnts.T))
        self.assertNotEqual(abs(Quaternion(*quats.T[1])), 0)
