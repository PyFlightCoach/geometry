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
from geometry import Point, Points
from geometry.point import Point, vector_projection
import numpy as np

class Line:
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    @staticmethod
    def fit_points(points: Points): 
        xy = np.polyfit(points.x, points.y, 1)
        xz = np.polyfit(points.x, points.z, 1)
        x0=min(points.x)
        x1 = max(points.x)
        return Line(
            Point(x0, x0 * xy[0] + xy[1], x0 * xz[0] + xz[1]),
            Point(x1, x1 * xy[0] + xy[1], x1 * xz[0] + xz[1])
            )
    
    def vector(self):
        return self.end - self.start
    

    def __str__(self):
        return "Line: start={}, stop={}".format(self.start, self.end)


    def project_point(self, point: Point) -> Point:
        return vector_projection(point - self.start, self.vector()) + self.start

    def parametric_value(self, point: Point):
        return abs(self.project_point(point) - self.start) / abs(self.vector())

    

