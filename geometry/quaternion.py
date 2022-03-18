"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
from .point import Point
from .base import Base

from typing import Union, Tuple
import numpy as np
from warnings import warn


class Quaternion(Base):
    cols=["w", "x", "y", "z"]

    @staticmethod
    def zero(count=1):
        return Quaternion(np.tile([count,0,0,0], (count,1)))

    @property
    def xyzw(self):
        return np.array([self.x, self.y, self.z, self.w]).T

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
                if len(self) == 1:
                    return Quaternion.full(self, len(other)) * other
                elif len(other) == 1:
                    return self * Quaternion.full(other, len(self))
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
                if len(other) == len(self.cols):
                    return self * Quaternion(self._dprep(other))

        raise NotImplementedError(f"{other.__class__.__name__} {other} cannot be multiplied by this Quaternion")
        
    def transform_point(self, point: Point):
        '''Transform a point by the rotation described by self'''

        if len(point) == len(self):
            qdata = np.column_stack((np.zeros(len(self)), point.data))
            return (self * Quaternion(qdata) * self.inverse()).axis

        if len(point) == 1 and len(self) > 1:
            return self.transform_point(Point.full(point, len(self))) 


        if len(self) == 1 and len(point) > 1:
            return Quaternion.full(self, len(point)).transform_point(point)

        else:
            raise ValueError()


    @staticmethod
    def from_euler(eul: Union[Point, Tuple[float, float, float]]):
        if isinstance(eul, tuple):
            eul = Point(*eul)
        # xyz-fixed Euler angle convention: matches ArduPilot AP_Math/Quaternion::from_euler
        half = eul * 0.5
        c = half.cos
        s = half.sin
        return Quaternion(
            w=c.y * c.z * c.x + s.y * s.z * s.x,
            x=c.y * c.z * s.x - s.y * s.z * c.x,
            y=s.y * c.z * c.x + c.y * s.z * s.x,
            z=c.y * s.z * c.x - s.y * c.z * s.x
        )


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
        return Point(np.array([roll, pitch, yaw]).T)

    @staticmethod
    def from_axis_angle(axangles: Point):
        small = 1e-6
        angles = abs(axangles)

        qdat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(angles), 1))

        if angles.any() >= small:
            baxangles = Point(axangles.data[angles >= small])
            bangles = angles[angles >= small]

            s = np.sin(bangles/2)
            c = np.cos(bangles/2)
            axis = baxangles / bangles

            qdat[angles >= small] = np.array([
                c, axis.x * s, axis.y * s, axis.z * s
            ]).T

        #qdat[abs(Quaternions(qdat)) < .001] = np.array([[1, 0, 0, 0]])
        return Quaternion(qdat)

    def to_axis_angle(self):
        """to a point of axis angles. must be normalized first."""
        angle = 2 * np.arccos(self.w)
        s = np.sqrt(1 - self.w**2)
        s[s < 1e-6] = 1.0
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

    def diff(self, dt: np.array) -> Point:
        """differentiate in the world frame"""
        assert len(dt) == len(self)
        dt = dt * len(dt) / (len(dt) - 1)

        ps = Quaternion.axis_rates(
            Quaternion(self.data[:-1, :]),
            Quaternion(self.data[1:, :])
        ) / dt[:-1]
        return Point(np.vstack([ps.data, ps.data[-1,:]]))#.remove_outliers(2)  # Bodge to get rid of phase jump

    def body_diff(self, dt: np.array) -> Point:
        """differentiate in the body frame"""
        assert len(dt) == len(self)
        dt = dt * len(dt) / (len(dt) - 1)

        ps = Quaternion.body_axis_rates(
            Quaternion(self.data[:-1, :]),
            Quaternion(self.data[1:, :])
        ) / dt[:-1]
        return Point(np.vstack([ps.data, ps.data[-1,:]]))#.remove_outliers(2)  # Bodge to get rid of phase jump

    
    def to_rotation_matrix(self):
        """http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        https://github.com/mortlind/pymath3d/blob/master/math3d/quaternion.py
        """
        n = self.norm()
        s, x, y, z = n.w, n.x, n.y, n.z
        x2, y2, z2 = n.x**2, n.y**2, n.z**2
        return np.array([
            [1 - 2 * (y2 + z2), 2 * x * y - 2 * s * z, 2 * s * y + 2 * x * z],
            [2 * x * y + 2 * s * z, 1 - 2 * (x2 + z2), -2 * s * x + 2 * y * z],
            [-2 * s * y + 2 * x * z, 2 * s * x + 2 * y * z, 1 - 2 * (x2 + y2)]
        ])

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
        q *= 0.5 / np.sqrt(t)
        return Quaternion(*q)

    def __str__(self):
        return "W:{w:.2f}\nX:{x:.2f}\nY:{y:.2f}\nZ:{z:.2f}".format(w=self.w, x=self.x, y=self.y, z=self.z)


def Q0(count=1):
    return Quaternion.zero(count)



def Quaternions(*args, **kwargs):
    warn("Quaternions is deprecated, you can now just use Quaternion", DeprecationWarning)
    return Quaternions(*args, **kwargs)