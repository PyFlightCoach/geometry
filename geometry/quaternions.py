from typing import Union
from geometry.quaternion import Quaternion
from numbers import Number
from geometry import Point, Points

import numpy as np
import pandas as pd


class Quaternions():
    def __init__(self, data):
        """Args: data (np.array): npoint * 4 array of point locations"""
        self.data = data

    @property
    def w(self):
        return self.data[:, 0]

    @property
    def x(self):
        return self.data[:, 1]

    @property
    def y(self):
        return self.data[:, 2]

    @property
    def z(self):
        return self.data[:, 3]

    @property
    def xyzw(self):
        return np.array([self.x, self.y, self.z, self.w]).T

    def __getitem__(self, index):
        return Quaternion(*list(self.data[index, :]))

    @staticmethod
    def from_pandas(df):
        return Quaternions(np.array(df))

    def to_pandas(self, prefix='', suffix='', columns=['w', 'x', 'y', 'z']):
        return pd.DataFrame(self.data, columns=[prefix + col + suffix for col in columns])

    def __abs__(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @property
    def axis(self):
        return Points(self.data[:, 1:])

    def norm(self):
        dab = 1 / abs(self)
        return self * dab

    def conjugate(self):

        return Quaternions(self.data * np.array([1, -1, -1, -1]))

    def inverse(self):
        return self.conjugate().norm()

    @property
    def count(self):
        return self.data.shape[0]

    def __mul__(self, other):
        if isinstance(other, Quaternions):
            pass
            if not self.count == other.count:
                return NotImplemented
            else:
                w = self.w * other.w - self.axis.dot(other.axis)
#
                xyz = self.w * other.axis + other.w * self.axis + \
                    self.axis.cross(other.axis)

                return Quaternions(np.column_stack([w, xyz.data]))

        elif isinstance(other, Number):
            return Quaternions(self.data * other)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Quaternions(self.data * other[:, np.newaxis])
                else:
                    return NotImplemented
            else:
                return NotImplemented
        elif isinstance(other, Quaternion):
            return self * Quaternions.from_quaternion(other, self.count)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Quaternions):
            return NotImplemented('this should have gone to __mul__')
        elif isinstance(other, float):
            return Quaternions(self.data * other)
        elif isinstance(other, Quaternion):
            return Quaternions.from_quaternion(other, self.count) * self
        else:
            return NotImplemented

    @staticmethod
    def from_quaternion(quat: Quaternion, count: int):
        return Quaternions(np.tile(list(quat), (count, 1)))

    @staticmethod
    def from_euler(eul: Points):
        half = eul * 0.5

        c = half.cosines()
        s = half.sines()

        return Quaternions(
            np.array([
                c.y * c.z * c.x + s.y * s.z * s.x,
                c.y * c.z * s.x - s.y * s.z * c.x,
                s.y * c.z * c.x + c.y * s.z * s.x,
                c.y * s.z * c.x - s.y * c.z * s.x
            ]).T
        )
    def to_euler(self):
        return Points(np.apply_along_axis(lambda row: Quaternions._to_euler(Quaternion(*row)), 1, self.data))

    @staticmethod
    def _to_euler(q1):
        #https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
        test = q1.x*q1.y + q1.z*q1.w
        if (test > 0.499):# { // singularity at north pole
            heading = 2 * np.arctan2(q1.x,q1.w)
            attitude = np.pi/2
            bank = 0
            return bank, heading, attitude
        elif (test < -0.499):# { // singularity at south pole
            heading = -2 * np.arctan2(q1.x,q1.w)
            attitude = - np.pi/2
            bank = 0
            return bank, heading, attitude
        
        sqx = q1.x*q1.x
        sqy = q1.y*q1.y
        sqz = q1.z*q1.z
        heading = np.arctan2(2*q1.y*q1.w-2*q1.x*q1.z , 1 - 2*sqy - 2*sqz)
        attitude = np.arcsin(2*test)
        bank = np.arctan2(2*q1.x*q1.w-2*q1.y*q1.z , 1 - 2*sqx - 2*sqz)
        return bank, heading, attitude

    def to_euler(self):
        # roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(sinp)
                
        # yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        test = np.abs(sinp) >= 0.9999
        if len(sinp[test]) > 0:
            pitch[test] = np.copysign(np.pi / 2, sinp[test])
            yaw[test] = np.zeros(len(sinp[test]))

            roll[test] = 2* np.arctan2(self.x[test],self.w[test])
        return Points(np.array([roll, pitch, yaw]).T)
    


    def transform_point(self, point: Union[Point, Points]):
        '''Transform a point by the rotation described by self'''
        if isinstance(point, Point):
            qdata = np.tile([0] + list(point), (self.count, 1))

            return (self * Quaternions(qdata) * self.inverse()).axis
        elif isinstance(point, Points):
            if point.count == self.count:

                qdata = np.column_stack((np.zeros(self.count), point.data))

                return (self * Quaternions(qdata) * self.inverse()).axis
            return NotImplemented
        else:
            return NotImplemented

    @staticmethod
    def from_axis_angle(axangles: Points):
        small = 0.000001
        angles = abs(axangles)

        qdat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(angles), 1))

        if angles.any() >= small:
            baxangles = Points(axangles.data[angles >= small])
            bangles = angles[angles >= small]

            s = np.sin(bangles/2)
            c = np.cos(bangles/2)
            axis = baxangles / bangles

            qdat[angles >= small] = np.array([
                c, axis.x * s, axis.y * s, axis.z * s
            ]).T

        #qdat[abs(Quaternions(qdat)) < .001] = np.array([[1, 0, 0, 0]])
        return Quaternions(qdat)

    def to_axis_angle(self):
        """to a point of axis angles. must be normalized first."""
        angle = 2 * np.arccos(self.w)
        s = np.sqrt(1 - self.w**2)
        s[s < 0.000001] = 1.0
        return self.axis * angle / s

    @staticmethod
    def axis_rates(q, qdot):
        wdash = qdot * q.conjugate()
        return wdash.norm().to_axis_angle() 

    @staticmethod
    def body_axis_rates(q, qdot):
        wdash = q.conjugate() * qdot
        return wdash.norm().to_axis_angle() 

    def rotate(self, rate: Points):
        return (Quaternions.from_axis_angle(rate) * self).norm()

    def body_rotate(self, rate: Points):
        return (self * Quaternions.from_axis_angle(rate)).norm()

    def diff(self, dt: np.array) -> Points:
        """differentiate in the world frame"""
        dt = dt * len(dt) / (len(dt) - 1)

        ps = Quaternions.axis_rates(
            Quaternions(self.data[:-1, :]),
            Quaternions(self.data[1:, :])
        ) / dt[:-1]
        return Points(np.vstack([ps.data, ps.data[-1,:]]))#.remove_outliers(2)  # Bodge to get rid of phase jump

    def body_diff(self, dt: np.array) -> Points:
        """differentiate in the body frame"""
        dt = dt * len(dt) / (len(dt) - 1)

        ps = Quaternions.body_axis_rates(
            Quaternions(self.data[:-1, :]),
            Quaternions(self.data[1:, :])
        ) / dt[:-1]
        return Points(np.vstack([ps.data, ps.data[-1,:]]))#.remove_outliers(2)  # Bodge to get rid of phase jump
