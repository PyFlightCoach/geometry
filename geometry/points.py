from geometry.point import Point
import re
import numpy as np
import pandas as pd
from numbers import Number
from scipy.cluster.vq import whiten

class Points(object):
    __array_priority__ = 15.0
    def __init__(self, data: np.array):
        """
        Args:
            data (np.array): npoint * 3 array of point locations
        """
        self.data = data

    @staticmethod
    def from_pandas(df):
        return Points(np.array(df))

    def to_pandas(self, prefix='', suffix='', columns=['x', 'y', 'z'], index=None):
        return pd.DataFrame(
            self.data, 
            columns=[prefix + col + suffix for col in columns],
            index=index
        )

    @property
    def x(self):
        return self.data[:,0]

    @property
    def y(self):
        return self.data[:,1]

    @property
    def z(self):
        return self.data[:,2]


    @property
    def count(self):
        return self.data.shape[0]  

    def __getitem__(self, index):
        return Point(*list(self.data[index, :]))

    def __abs__(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def abs(self):
        return Points(abs(self.data))

    def __add__(self, other):
        if isinstance(other, Points):
            return Points(self.data + other.data)
        elif isinstance(other, Point):
            return Points(self.data + Points.from_point(other, self.count).data)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Points):
            return Points(self.data - other.data)
        elif isinstance(other, Point):
            return Points(self.data - Points.from_point(other, self.count).data)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Point):
            return Points(Points.from_point(other, self.count).data - self.data)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


    @staticmethod
    def from_point(point, count):
        return Points.full(point,count)

    @staticmethod
    def full(point: Point, count: int):
        return Points(np.tile(list(point), (count, 1)))

    def __mul__(self, other):
        if isinstance(other, Points):
            return Points(self.data * other.data)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Points(self.data * other[:, np.newaxis])
                else:
                    raise NotImplementedError("this will return an unexpected result")
            else:
                return NotImplemented
        elif isinstance(other, Number):
            return Points(self.data * other)
        elif isinstance(other, Point):
            return Points(self.data * list(other))
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Points):
            return Points(self.data / other.data)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == self.count:
                    return Points(self.data / other[:, np.newaxis])
                else:
                    return NotImplemented
            else:
                return NotImplemented
        elif isinstance(other, Number):
            return Points(self.data / other)

    def __neg__(self):
        return -1 * self

    def scale(self, value):
        if isinstance(value, Number):
            return (value / abs(self)) * self
        else:
            return NotImplemented

    def unit(self):
        return self.scale(1)

    def sines(self):
        return Points(np.sin(self.data))
    
    def cosines(self):
        return Points(np.cos(self.data))

    def acosines(self):
        return Points(np.acos(self.data))

    def asines(self):
        return Points(np.asin(self.data))

    def dot(self, other):
        return np.einsum('ij,ij->i', self.data, other.data)

    def cross(self, other):
        return Points(np.cross(self.data, other.data))

    def diff(self, dt:np.array):
        return Points(np.gradient(self.data,axis=0) / np.tile(dt, (3,1)).T)

    def remove_outliers(self, nstds = 2):
        ab = abs(self)
        std = np.nanstd(ab)
        mean = np.nanmean(ab)

        data = self.data.copy()

        data[abs(ab - mean) > nstds * std, :] = [np.nan, np.nan, np.nan]

        return Points(pd.DataFrame(data).fillna(method="ffill").to_numpy())

    def mean(self):
        return Point(self.x.mean(), self.y.mean(), self.z.mean())

    def max(self):
        return Point(self.x.max(), self.y.max(), self.z.max())

    def min(self):
        return Point(self.x.min(), self.y.min(), self.z.min())

    def norm(self, mode="elements"):
        if mode == "elements":
            return self / abs(self).max()
        elif mode == "full":
            return self / max(list(abs(self).max()))

    def whiten(self):
        return Points(whiten(self.data))

    @staticmethod
    def X(value):
        return Points(np.array([value, np.zeros(value.shape), np.zeros(value.shape)]).T)
    @staticmethod
    def Y(value):
        return Points(np.array([np.zeros(value.shape), value, np.zeros(value.shape)]).T)
    @staticmethod
    def Z(value):
        return Points(np.array([np.zeros(value.shape), np.zeros(value.shape), value]).T)