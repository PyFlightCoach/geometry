from geometry.factory import gfactory
import numpy as np
from typing import Union, List


class Points:
    pass

class Point(Points):
    pass

Points, Point = gfactory("Point", ["x", "y", "z"])


def pstaticmethod(meth):
    setattr(Points, meth.__name__, staticmethod(meth))

def pmethod(meth):

    def pretmeth(*args, **kwargs):
        res = meth(*args, **kwargs)
        if isinstance(res, Points):
            if res.count==1:
                res = Point(res.data)
        return res

    setattr(Points, meth.__name__, pretmeth)

import geometry.point.instance_methods 




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
    angle = angle_between(p1, p2) % np.pi
    return min(angle, np.pi - angle)


def angle_between(p1: Point, p2: Point):
    raisezero([p1, p2])
    return np.arccos(cos_angle_between(p1, p2))


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