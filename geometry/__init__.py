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
from .point import Point, cross_product, dot_product, cos_angle_between, scalar_projection,vector_projection
from .points import Points
from .gps import GPSPosition
from .coordinate_frame import Coord
from .quaternion import Quaternion
from .quaternions import Quaternions
from .transformation import Transformation
from .line import Line


def Euler(x,y,z):
    return Quaternion.from_euler(Point(x,y,z))