from geometry import Base, Point
import numpy as np


class Mass(Base):
    cols = ["m", "xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]

    @staticmethod
    def from_matrix(m, M):
        return Mass(m, *M.reshape(9))
    
    @staticmethod
    def from_principal(m, xx, yy, zz):
        return Mass.from_matrix(m, [xx,yy,zz] * np.identity(3))
    
    @staticmethod
    def from_double_symmetry(m, xxyyzz):
        return Mass.from_matrix(m, xxyyzz * np.identity(3))

    @staticmethod
    def point(m):
        return Mass(m, *np.zeros(9))

    @staticmethod
    def sphere(m, r):
        return Mass.from_double_symmetry(m, (2/5) * m * r**2 )

    @staticmethod
    def cuboid(m, lx, ly, lz):
        cf = lambda y, z: (1/12) * m * (y**2 + z**2)
        return Mass.from_principal(m,cf(ly, lz), cf(lz, lx), cf(lx, ly))

    def matrix(self):
        return self.data[:,1:].reshape((len(self), 3, 3))

    @property
    def I(self):
        return self.matrix()

    def offset(self, v: Point):
        xx = v.y**2 + v.z**2
        yy = v.z**2 + v.x**2
        zz = v.x**2 + v.y**2
        xy = v.x * v.y
        xz = v.x * v.z
        yz = v.y * v.z
        return Mass(
            *(self.m * np.array([
                np.zeros(len(v)), xx, -xy, -xz, -xy, yy, -yz, -xz, -yz, zz
            ])) 
        ) + self


    