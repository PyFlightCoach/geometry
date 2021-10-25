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

from . import Point, cross_product
from typing import List
import numpy as np
import pandas as pd

# TODO look at scipy.spatial.transform.Rotation


class Coord(object):
    __slots__ = ["origin", "x_axis", "y_axis", "z_axis",
                 "_rotation_matrix", "_inverse_rotation_matrix"]

    def __init__(self, origin: Point, x_axis: Point, y_axis: Point, z_axis: Point):
        self.origin = origin
        self.x_axis = x_axis.unit()
        self.y_axis = y_axis.unit()
        self.z_axis = z_axis.unit()

    def __iter__(self):
        for i in list(self.origin) + list(self.x_axis) + list(self.z_axis):
            yield i

    def __str__(self):
        return "origin:{0}\nx_axis:{1}\ny_axis:{1}\nz_axis:{2}".format(str(self.origin), str(self.x_axis), str(self.y_axis), str(self.z_axis))

    def to_dict(self):
        return dict(
            origin=self.origin.to_dict(),
            x_axis=self.x_axis.to_dict(),
            y_axis=self.y_axis.to_dict(),
            z_axis=self.z_axis.to_dict()
        )
    @property
    def axes(self):
        return [self.x_axis, self.y_axis, self.z_axis]

    @staticmethod
    def from_nothing():
        return Coord(Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0), Point(0, 0, 1))

    @staticmethod
    def from_xy(origin: Point, x_axis: Point, y_axis: Point):
        z_axis = cross_product(x_axis, y_axis)
        return Coord(origin, x_axis, cross_product(z_axis, x_axis), z_axis)

    @staticmethod
    def from_yz(origin: Point, y_axis: Point, z_axis: Point):
        x_axis = cross_product(y_axis, z_axis)
        return Coord(origin, x_axis, y_axis, cross_product(x_axis, y_axis))

    @staticmethod
    def from_zx(origin: Point, z_axis: Point, x_axis: Point):
        y_axis = cross_product(z_axis, x_axis)
        return Coord(origin, cross_product(y_axis, z_axis), y_axis, z_axis)

    def euler_rotation(self, angles: Point = Point(0, 0, 0)):
        return self.rotate(angles.to_rotation_matrix())

    @property
    def rotation_matrix(self):
        return np.array([self.x_axis.to_list(), self.y_axis.to_list(), self.z_axis.to_list()])

    @property
    def inverse_rotation_matrix(self):
        return np.array([
            [self.x_axis.x, self.y_axis.x, self.z_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z]
        ])

    def rotate(self, rotation_matrix=List[List[float]]):
        return Coord(
            origin=self.origin,
            x_axis=self.x_axis.rotate(rotation_matrix),
            y_axis=self.y_axis.rotate(rotation_matrix),
            z_axis=self.z_axis.rotate(rotation_matrix)
        )

    def __eq__(self, other):
        return \
            self.origin == other.origin and \
            self.x_axis == other.x_axis and \
            self.y_axis == other.y_axis and \
            self.z_axis == other.z_axis

    def translate(self, point):
        return Coord(self.origin + point, self.x_axis, self.y_axis, self.z_axis)

    def get_plot_df(self, length=10):
        def make_ax(ax: Point, colour: str):
            return [
                list(self.origin) + [colour],
                list(self.origin + ax * length) + [colour],
                list(self.origin) + [colour]
            ]

        axes = []
        for ax, col in zip(self.axes, ['red', 'blue', 'green']):
            axes += make_ax(ax, col)

        return pd.DataFrame(
            axes,
            columns=list('xyzc')
        )
