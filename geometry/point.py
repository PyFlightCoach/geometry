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
from math import pi, sqrt, acos
import numpy as np
from typing import List
from numbers import Number
from typing import List, Union, Dict


class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        for i in [self.x, self.y, self.z]:
            yield i

    def to_list(self):
        """can be deprecated, instead use list()"""
        return [self.x, self.y, self.z]

    def to_tuple(self):
        """can be deprecated, instead use tuple()"""
        return(self.x, self.y, self.z)

    def to_dict(self, prefix=''):
        return {prefix+'x': self.x, prefix+'y': self.y, prefix+'z': self.z}

    @staticmethod
    def from_dict(value: dict):
        value = {key[-1]: value for key, value in value.items()}
        return Point(value['x'], value['y'], value['z'])

    def __str__(self):
        return "X:{x:.2f},Y:{y:.2f},Z:{z:.2f}".format(x=self.x, y=self.y, z=self.z)

    def __abs__(self):
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(
                x=self.x + other.x,
                y=self.y + other.y,
                z=self.z + other.z
            )
        elif isinstance(other, Number):
            return Point(
                x=self.x + other,
                y=self.y + other,
                z=self.z + other
            )
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(
                x=self.x - other.x,
                y=self.y - other.y,
                z=self.z - other.z
            )
        elif isinstance(other, Number):
            return Point(
                x=self.x - other,
                y=self.y - other,
                z=self.z - other
            )
        else:
            return NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Point):
            return (other.x == self.x) and (other.y == self.y) and (other.z == self.z)
        else:
            try:
                return (other == self.x) and (other == self.y) and (other == self.z)
            finally:
                return False

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(x=other.x * self.x, y=other.y * self.y, z=other.z * self.z)
        elif isinstance(other, Number):
            return Point(x=other * self.x, y=other * self.y, z=other * self.z)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Point) or isinstance(other, Number):
            return self.__mul__(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Point):
            return Point(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, Number):
            return Point(self.x / other, self.y / other, self.z / other)
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Point):
            return abs(self) < abs(other)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Point):
            return abs(self) > abs(other)
        else:
            return NotImplemented

    def __neg__(self):
        return -1 * self

    def scale(self, value):
        return self.__mul__(value / self.__abs__())

    def is_equal(self, tolerance=0):
        if abs(self.x - self.y) > tolerance:
            return False
        if abs(self.x - self.z) > tolerance:
            return False
        return True

    def unit(self):
        return self.scale(1)

    @property
    def cosines(self):
        return Point(x=np.cos(self.x), y=np.cos(self.y), z=np.cos(self.z))

    @property
    def sines(self):
        return Point(x=np.sin(self.x), y=np.sin(self.y), z=np.sin(self.z))

    @property
    def acosines(self):
        return Point(x=np.arccos(self.x), y=np.arccos(self.y), z=np.arccos(self.z))

    @property
    def asines(self):
        return Point(x=np.arcsin(self.x), y=np.arcsin(self.y), z=np.arcsin(self.z))

    def rotate(self, rotation_matrix=List[List[float]]):
        return Point(
            self.x * rotation_matrix[0][0] + self.y *
            rotation_matrix[0][1] + self.z * rotation_matrix[0][2],
            self.x * rotation_matrix[1][0] + self.y *
            rotation_matrix[1][1] + self.z * rotation_matrix[1][2],
            self.x * rotation_matrix[2][0] + self.y *
            rotation_matrix[2][1] + self.z * rotation_matrix[2][2],
        )

    def unit_cost(self, other):
        return abs(self - other)

    def to_rotation_matrix(self):
        '''returns the rotation matrix based on apoint representing Euler angles'''
        s = self.sines
        c = self.cosines
        return [
            [c.z * c.y, c.z * s.y * s.x - c.x * s.z, c.x * c.z * s.y + s.x * s.z],
            [c.y * s.z, c.x * c.z + s.x * s.y *
                s.z, -1 * c.z * s.x + c.x * s.y * s.z],
            [-1 * s.y, c.y * s.x, c.x * c.y]
        ]

    @staticmethod
    def X(mag: float = 1.0):
        return Point(mag, 0.0, 0.0)
        
    @staticmethod
    def Y(mag: float = 1.0):
        return Point(0.0, mag, 0.0)

    @staticmethod
    def Z(mag: float = 1.0):
        return Point(0.0, 0.0, mag)

    @staticmethod
    def XYZ(mag: float = 0.0):
        return Point(mag, mag, mag)

    @staticmethod
    def zeros():
        return Point(0.0,0.0,0.0)

def dot_product(p1: Point, p2: Point):
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z


def cos_angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    return dot_product(p1.unit(), p2.unit())


def cross_product(p1: Point, p2: Point) -> Point:
    return Point(
        x=p1.y * p2.z - p1.z * p2.y,
        y=p1.z * p2.x - p1.x * p2.z,
        z=p1.x * p2.y - p1.y * p2.x
    )


def scalar_projection(from_vec: Point, to_vec: Point):
    try:
        return cos_angle_between(from_vec, to_vec) * abs(from_vec)
    except ValueError:
        return 0


def vector_projection(from_vec: Point, to_vec: Point) -> Point:
    if abs(from_vec) == 0:
        return Point(0, 0, 0)
    return to_vec.scale(scalar_projection(from_vec, to_vec))


def is_parallel(p1: Point, p2: Point, tolerance=0.000001):
    raisezero([p1, p2])
    if p1 == p2:
        return True
    return abs(abs(cos_angle_between(p1, p2)) - 1) < tolerance


def is_anti_parallel(p1: Point, p2: Point, tolerance=0.000001):
    raisezero([p1, p2])
    if p1 == - p2:
        return True
    return abs(cos_angle_between(p1, p2) + 1) < tolerance


def is_perpendicular(p1: Point, p2: Point, tolerance=0.000001):
    raisezero([p1, p2])
    return abs(dot_product(p1, p2)) < tolerance


def min_angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    angle = angle_between(p1, p2) % pi
    return min(angle, pi - angle)


def angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    return acos(cos_angle_between(p1, p2))


def arbitrary_perpendicular(v: Point) -> Point:
    raisezero(v)
    if v.x == 0 and v.y == 0:
        return Point(0, 1, 0)
    return Point(-v.y, v.x, 0).unit


def raisezero(points: Union[Point, List[Point]]):
    if isinstance(points, Point):
        _raisezero(points)
    else:
        for point in points:
            _raisezero(point)


def _raisezero(point: Point, tolerance=0.000001):
    if abs(point) < tolerance:
        raise ValueError('magnitude less than tolerance')

def vector_norm(point: Point):
    return abs(point)

def normalize_vector(point: Point):
    raisezero(point)
    return point / abs(point)

