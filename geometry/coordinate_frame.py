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

from geometry import Point, Quaternion, PX, PY, PZ, P0
from typing import List
import numpy as np
import pandas as pd
from geometry.base import Base
# TODO look at scipy.spatial.transform.Rotation


class Coord(Base):
    cols = [
        "ox", "oy", "ox",
        "x1", "y1", "z1",
        "x2", "y2", "z2",
        "x3", "y3", "z3",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin=Point(self.data[:,:3])
        self.x_axis=Point(self.data[:,3:6])
        self.y_axis=Point(self.data[:,6:9])
        self.z_axis=Point(self.data[:,9:12])
    
    @staticmethod
    def from_axes(o:Point, x:Point, y:Point, z:Point):
        return Coord(np.concatenate([
            o.data,
            x.unit().data,
            y.unit().data,
            z.unit().data
        ],axis=1))

    @staticmethod
    def from_nothing(count=1):
        return Coord(P0(count), PX(count), PY(count), PZ(count))

    @staticmethod
    def from_xy(origin: Point, x_axis: Point, y_axis: Point):
        assert len(origin) == len(x_axis) == len(y_axis)
        z_axis = x_axis.cross(y_axis)
        return Coord(origin, x_axis, z_axis.cross(x_axis), z_axis)

    @staticmethod
    def from_yz(origin: Point, y_axis: Point, z_axis: Point):
        assert len(origin) == len(y_axis) == len(z_axis)
        x_axis = y_axis.cross(z_axis)
        return Coord(origin, x_axis, y_axis, x_axis.cross(y_axis))

    @staticmethod
    def from_zx(origin: Point, z_axis: Point, x_axis: Point):
        assert len(origin) == len(z_axis) == len(x_axis)
        y_axis = z_axis.cross(x_axis)
        return Coord(origin, y_axis.cross(z_axis), y_axis, z_axis)

    def rotation_matrix(self):
        return np.array([
            self.x_axis.data, 
            self.y_axis.data, 
            self.z_axis.data
        ])


    def inverse_rotation_matrix(self):
        return np.array([
            [self.x_axis.x, self.y_axis.x, self.z_axis.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z]
        ])

    def rotate(self, rotation_matrix=np.ndarray):
        assert rotation_matrix.shape[1:] == (3, 3)
        assert rotation_matrix.shape[0] == len(self.origin) or rotation_matrix.shape[0] == 1

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
                self.origin.data + [colour],
                self.origin.data + ax * length + [colour],
                self.origin.data + [colour]
            ]

        axes = []
        for ax, col in zip(self.axes, ['red', 'blue', 'green']):
            axes += make_ax(ax, col)

        return pd.DataFrame(
            axes,
            columns=list('xyzc')
        )
