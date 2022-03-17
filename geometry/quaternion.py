
from .point import Point
from .base import Base

from math import atan2, asin, copysign, pi, sqrt
from typing import List, Dict, Union, Tuple
import numpy as np


class Quaternion(Base):
    cols=["w", "x", "y", "z"]

    @staticmethod
    def zero():
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @property
    def xyzw(self):
        return [self.x, self.y, self.z, self.w]

    @property
    def axis(self):
        return Point(self.data[:,1:])

    def norm(self):
        return self / abs(self)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        return self.conjugate().norm()

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            if not len(self) == len(other):
                pass
            else:
                w = self.w * other.w - self.axis.dot(other.axis)
#
                xyz = self.w * other.axis + other.w * self.axis + \
                    self.axis.cross(other.axis)

                return Quaternion(np.column_stack([w, xyz.data]))
        elif isinstance(other, float) or isinstance(other, int):
            return Quaternion(self.data * other)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Quaternion(self.data * other[:, np.newaxis])
        elif isinstance(other, Quaternion):
            return self * Quaternion.full(other, self.count)
        raise NotImplementedError(f"Not implemented for {other.__class__.__name__} yet")
        
    def transform_point(self, point: Point):
        '''Transform a point by the rotation described by self'''
        if isinstance(point, Point):
            return (self * Quaternion(*[0] + list(point)) * self.inverse()).axis
        else:
            return NotImplemented

    @staticmethod
    def from_euler(eul: Union[Point, Tuple[float, float, float]]):
        if isinstance(eul, tuple):
            eul = Point(*eul)
        # xyz-fixed Euler angle convention: matches ArduPilot AP_Math/Quaternion::from_euler
        half = eul * 0.5
        c = half.cosines
        s = half.sines
        return Quaternion(
            w=c.y * c.z * c.x + s.y * s.z * s.x,
            x=c.y * c.z * s.x - s.y * s.z * c.x,
            y=s.y * c.z * c.x + c.y * s.z * s.x,
            z=c.y * s.z * c.x - s.y * c.z * s.x

        )

    @staticmethod
    def from_axis_angle(axangle: Point):
        angle = abs(axangle)
        if (angle < .001):
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        axis = axangle / angle
        s = np.sin(angle/2)
        c = np.cos(angle/2)
        return Quaternion(c, axis.x * s, axis.y * s, axis.z * s)

    # from https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
    def to_axis_angle(self):
        """to a point of axis angles. must be normalized first."""
        if (self.w > 1):
            self.norm()
        angle = 2*np.arccos(self.w)
        s = np.sqrt(1 - self.w**2)
        if (s < .001):
            return self.axis * angle
        else:
            return self.axis * angle / s

    @staticmethod
    def axis_rates(q, qdot):
        wdash = qdot * q.conjugate()
        return wdash.norm().to_axis_angle()

    @staticmethod
    def body_axis_rates(q, qdot):
        wdash = q.conjugate() * qdot
        return wdash.norm().to_axis_angle()

    def rotate(self, rate: Point):
        return (Quaternion.from_axis_angle(rate) * self).norm()

    def body_rotate(self, rate: Point):
        return (self * Quaternion.from_axis_angle(rate)).norm()

    def to_euler(self):
        # xyz-fixed Euler angle convention: matches ArduPilot AP_Math/Quaternion::to_euler
        roll = atan2(
            2 * (self.w * self.x + self.y * self.z),
            1 - 2 * (self.x * self.x + self.y * self.y)
        )

        _sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(_sinp) >= 1:
            pitch = copysign(pi / 2, _sinp)
        else:
            pitch = asin(_sinp)

        yaw = atan2(
            2 * (self.w * self.z + self.x * self.y),
            1 - 2 * (self.y * self.y + self.z * self.z)
        )

        return Point(roll, pitch, yaw)

    def to_rotation_matrix(self):
        """http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        https://github.com/mortlind/pymath3d/blob/master/math3d/quaternion.py
        """
        n = self.norm()
        s, x, y, z = n.w, n.x, n.y, n.z
        x2, y2, z2 = n.x**2, n.y**2, n.z**2
        return [
            [1 - 2 * (y2 + z2), 2 * x * y - 2 * s * z, 2 * s * y + 2 * x * z],
            [2 * x * y + 2 * s * z, 1 - 2 * (x2 + z2), -2 * s * x + 2 * y * z],
            [-2 * s * y + 2 * x * z, 2 * s * x + 2 * y * z, 1 - 2 * (x2 + y2)]
        ]

    @staticmethod
    def from_rotation_matrix(matrix: np.ndarray):
        # This method assumes row-vector and postmultiplication of that vector
        m = matrix.conj().transpose()
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0] - m[0, 2], m[0, 1] +
                     m[1, 0], t, m[1, 2] + m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1] - m[1, 0], m[2, 0] +
                     m[0, 2], m[1, 2] + m[2, 1], t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]

        q = np.array(q).astype('float64')
        q *= 0.5 / sqrt(t)
        return Quaternion(*q)

    def __str__(self):
        return "W:{w:.2f}\nX:{x:.2f}\nY:{y:.2f}\nZ:{z:.2f}".format(w=self.w, x=self.x, y=self.y, z=self.z)

    @staticmethod
    def from_dict(value: Dict):
        return Quaternion(
            value['w'],
            value['x'],
            value['y'],
            value['z']
        )
