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
from .point import *
from .quaternion import *
from .gps import GPS
from .coordinate_frame import Coord
from .transformation import Transformation
from .mass import Mass


def Euler(*args, **kwargs) -> Quaternion:
    return Quaternion.from_euler(Point(*args, **kwargs))
    

def Euldeg(*args, **kwargs) -> Quaternion:
    return Quaternion.from_euler(Point(*args, **kwargs).radians())


def angle_diff(a, b): 
    d1 = a - b
    d2 = d1 - 2 * np.pi
    bd=np.abs(d2) < np.abs(d1)
    d1[bd] = d2[bd]
    d3 = d1 + 2 * np.pi
    bd=np.abs(d3) < np.abs(d1)
    d1[bd] = d3[bd]

    return d1