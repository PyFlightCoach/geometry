from geometry import Base, Point, Quaternion, Point, P0, Q0, Coord

import numpy as np
from typing import Union


class Transformation(Base):
    cols = ["x", "y", "z", "rw", "rx", "ry", "rz"]

    def __init__(self, *args, **kwargs):
        if len(args) == len(kwargs) == 0:
            args = np.concatenate([P0().data,Q0().data],axis=1)
        super().__init__(*args, **kwargs)
        self.p = Point(self.data[:,:3])
        self.q = Quaternion(self.data[:,3:])
    
    @staticmethod
    def build(p:Point, q:Quaternion):
        if len(p) == len(q):
            return Transformation(np.concatenate([
                p.data,
                q.data
            ],axis=1))
        elif len(p) == 1 and len(q) > 1:
            return Transformation.build(Point.full(p, len(q)), q)
        elif len(p) > 1 and len(q) >= 1:
            return Transformation.build(p, Point.full(q, len(p)))
        else:
            raise ValueError("incompatible lengths")

    @staticmethod
    def zero(count):
        return Transformation.build(P0(count), Q0(count))

    @property
    def translation(self) -> Point:
        return self.p

    @property
    def rotation(self) -> Quaternion:
        return self.q

    @staticmethod
    def from_coords(coord_a, coord_b):
        q1 = Quaternion.from_rotation_matrix(coord_b.rotation_matrix()).inverse()
        q2 = Quaternion.from_rotation_matrix(coord_a.rotation_matrix())
        return Transformation.build(
            coord_b.origin - coord_a.origin,
            -q1 * q2
        )

    def rotate(self, oin: Union[Point, Quaternion]):
        if isinstance(oin, Point):
            return self.q.transform_point(oin)
        elif isinstance(oin, Quaternion):
            return self.q * oin

    def translate(self, point: Point):
        return point + self.p

    def point(self, point: Point):
        return self.translate(self.rotate(point))       

    def coord(self, coord):
        return coord.translate(self.p).rotate(self.q.to_rotation_matrix())
