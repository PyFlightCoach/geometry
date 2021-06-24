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

    

