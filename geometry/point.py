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
from pprint import pp
from .base import Base
import numpy as np
import pandas as pd
from typing import List 
from warnings import warn


class Point(Base):
    cols=["x", "y", "z"]
    from_np = [
        "sin","cos","tan",
        "arcsin","arccos","arctan",
    ]

    def scale(self, value):
        a, b=value, abs(self)
        res = a/b
        res[b==0] = 0
        res = self * res
        
        return res
        
    def unit(self):
        return self.scale(1)

    def remove_outliers(self, nstds = 2):
        ab = abs(self)
        std = np.nanstd(ab)
        mean = np.nanmean(ab)

        data = self.data.copy()

        data[abs(ab - mean) > nstds * std, :] = [np.nan, np.nan, np.nan]

        return Point(pd.DataFrame(data).fillna(method="ffill").to_numpy())

    def mean(self):
        return Point(np.mean(self.data, axis=0))

    def max(self):
        return Point(np.max(self.data, axis=0))
    
    def min(self):
        return Point(np.min(self.data, axis=0))

    def angles(self, p2):
        return (self.cross(p2) / (abs(self) * abs(p2))).asines()
    
    def angle(self, p2):
        return abs(Point.angles(self, p2))
    
    @staticmethod
    def X(value=1, count=1):
        return Point(np.tile([value,0,0], (count, 1)))

    @staticmethod
    def Y(value=1, count=1):
        return Point(np.tile([0,value,0], (count, 1)))

    @staticmethod
    def Z(value=1, count=1):
        return Point(np.tile([0,0,value], (count, 1)))

    def rotate(self, rmat=np.ndarray):
        if len(rmat.shape) == 3:
            pass
        elif len(rmat.shape) == 2:
            rmat = np.reshape(rmat, (1, 3, 3 ))
        else:
            raise TypeError("expected a 3x3 matrix")
        
        return self.dot(rmat)

    def to_rotation_matrix(self):
        '''returns the rotation matrix based on a point representing Euler angles'''
        s = self.sin
        c = self.cos
        return np.array([
            [
                c.z * c.y, 
                c.z * s.y * s.x - c.x * s.z, 
                c.x * c.z * s.y + s.x * s.z
            ], [
                c.y * s.z, 
                c.x * c.z + s.x * s.y * s.z, 
                -1 * c.z * s.x + c.x * s.y * s.z
            ],
            [
                -1 * s.y, 
                c.y * s.x, 
                c.x * c.y
            ]
        ])

    @staticmethod
    def zeros(count=1):
        return Point(np.zeros((count,3)))

def Points(*args, **kwargs):
    warn("Points is deprecated, you can now just use Point", DeprecationWarning)
    return Point(*args, **kwargs)


def PX(length=1, count=1):
    return Point.X(length, count)

def PY(length=1, count=1):
    return Point.Y(length, count)

def PZ(length=1, count=1):
    return Point.Z(length, count)

def P0(count=1):
    return Point.zeros(count)

def ppmeth(func):
    def wrapper(a, b, *args, **kwargs):
        assert all([isinstance(arg, Point) for arg in args])
        assert len(a) == len(b) or len(a) == 1 or len(b) == 1
        return func(a, b, *args, **kwargs)

    setattr(Point, func.__name__, wrapper)
    return wrapper


@ppmeth
def cross(a, b) -> Point:
    return Point(
        a.y * b.z - a.z * b.y, 
        a.z * b.x - a.x * b.z, 
        a.x * b.y - a.y * b.x
    )


@ppmeth
def cos_angle_between(a: Point, b: Point) -> np.ndarray:
    if a == 0 or b == 0:
        raise ValueError("cannot measure the angle to a zero length vector")
    return a.unit().dot(b.unit())


@ppmeth
def angle_between(a: Point, b: Point) -> np.ndarray:
    return np.arccos(a.cos_angle_between(b))

@ppmeth
def scalar_projection(a: Point, b: Point) -> Point:
    if a==0 or b==0:
        return 0
    return a.cos_angle_between(b) * abs(a)

@ppmeth
def vector_projection(a: Point, b: Point) -> Point:
    if abs(a) == 0:
        return Point.zeros()
    return b.scale(a.scalar_projection(b))

@ppmeth
def is_parallel(a: Point, b: Point, tolerance=1e-6):
    if a.unit() == b.unit():
        return True
    return abs(a.cos_angle_between(b) - 1) < tolerance

@ppmeth
def is_perpendicular(a: Point, b: Point, tolerance=1e-6):
    return abs(a.dot(b)) < tolerance

@ppmeth
def min_angle_between(p1: Point, p2: Point):
    angle = angle_between(p1, p2) % np.pi
    return min(angle, np.pi - angle)

@ppmeth
def angle_between(a: Point, b: Point) -> float: 
    return np.arccos(cos_angle_between(a, b))

def arbitrary_perpendicular(v: Point) -> Point:
    if v.x == 0 and v.y == 0:
        return Point(0, 1, 0)
    return Point(-v.y, v.x, 0).unit

def vector_norm(point: Point):
    return abs(point)

def normalize_vector(point: Point):
    return point / abs(point)