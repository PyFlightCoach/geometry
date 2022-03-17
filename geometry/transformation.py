from geometry import Point, Quaternion, Point, P0, Q0, Coord

import numpy as np
from typing import Union


class Transformation:
    def __init__(self, p: Point=P0(), q: Quaternion=Q0()):
        assert len(p) == len(q)
        self.p = p
        self.q = q

    @property
    def translation(self) -> Point:
        return self.p

    @property
    def rotation(self) -> Quaternion:
        return self.q

    @staticmethod
    def from_coords(coord_a, coord_b):
        return Transformation(
            coord_b.origin - coord_a.origin,
            Quaternion.from_rotation_matrix(
                np.dot(
                    coord_b.inverse_rotation_matrix,
                    coord_a.rotation_matrix
                ))
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
        return coord.translate(self.p).rotate(
            self.q.to_rotation_matrix()
        )
