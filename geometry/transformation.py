from geometry import Point, Quaternion, Coord, Points, Quaternions

import numpy as np
from typing import Union


class Transformation():
    def __init__(self, translation: Point = Point(0.0, 0.0, 0.0), rotation: Quaternion = Quaternion(1, 0, 0, 0)):
        self.translation = translation
        self.rotation = rotation

    @staticmethod
    def from_coords(coord_a: Coord, coord_b: Coord):
        return Transformation(
            coord_b.origin - coord_a.origin,
            Quaternion.from_rotation_matrix(
                np.dot(
                    coord_b.inverse_rotation_matrix,
                    coord_a.rotation_matrix
                ))
        )

    def rotate(self, point: Union[Point, Points]):
        if isinstance(point, Point):
            return self.rotation.transform_point(point)
        elif isinstance(point, Points):
            return Quaternions.from_quaternion(self.rotation, point.count).transform_point(point)
        else:
            return NotImplemented

    def translate(self, point: Union[Point, Points]):
        return point + self.translation

    def point(self, point: Union[Point, Points]):
        return self.translate(self.rotate(point))

    def quat(self, quat: Union[Quaternion, Quaternions]):
        return self.rotation * quat

    def coord(self, coord: Coord = Coord.from_nothing()):
        return coord.translate(self.translation).rotate(
            self.rotation.to_rotation_matrix()
        )
