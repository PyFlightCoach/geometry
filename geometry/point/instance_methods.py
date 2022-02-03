from . import Points, Point, pmethod, pstaticmethod
import numpy as np
import pandas as pd
from numbers import Number
from scipy.cluster.vq import whiten

@pmethod
def __abs__(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

@pmethod
def __add__(self, other):
    if isinstance(other, Points):
        return Points(self.data + other.data)
    elif isinstance(other, Point):
        return Points(self.data + Points.from_point(other, self.count).data)
    else:
        return NotImplemented

@pmethod
def __radd__(self, other):
    return self.__add__(other)

@pmethod
def __sub__(self, other):
    if isinstance(other, Points):
        return Points(self.data - other.data)
    elif isinstance(other, Point):
        return Points(self.data - Points.from_point(other, self.count).data)
    else:
        return NotImplemented

@pmethod
def __rsub__(self, other):
    if isinstance(other, Point):
        return Points(Points.from_point(other, self.count).data - self.data)
    else:
        return NotImplemented

@pmethod
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

@pmethod
def __rmul__(self, other):
    return self.__mul__(other)

@pmethod
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

@pmethod
def __neg__(self):
    return -1 * self

@pmethod
def scale(self, value):
    if isinstance(value, Number):
        return (value / abs(self)) * self
    else:
        return NotImplemented

@pmethod
def unit(self):
    return self.scale(1)

@pmethod
def sines(self):
    return Points(np.sin(self.data))

@pmethod
def cosines(self):
    return Points(np.cos(self.data))

@pmethod
def acosines(self):
    return Points(np.arccos(self.data))

@pmethod
def asines(self):
    return Points(np.arcsin(self.data))

@pmethod
def dot(self, other):
    return np.einsum('ij,ij->i', self.data, other.data)

@pmethod
def cross(self, other):
    return Points(np.cross(self.data, other.data))

@pmethod
def diff(self, dt:np.array):
    return Points(np.gradient(self.data,axis=0) / np.tile(dt, (3,1)).T)

@pmethod
def remove_outliers(self, nstds = 2):
    ab = abs(self)
    std = np.nanstd(ab)
    mean = np.nanmean(ab)

    data = self.data.copy()

    data[abs(ab - mean) > nstds * std, :] = [np.nan, np.nan, np.nan]

    return Points(pd.DataFrame(data).fillna(method="ffill").to_numpy())

@pmethod
def mean(self):
    return Point(self.x.mean(), self.y.mean(), self.z.mean())

@pmethod
def max(self):
    return Point(self.x.max(), self.y.max(), self.z.max())

@pmethod
def min(self):
    return Point(self.x.min(), self.y.min(), self.z.min())

@pmethod
def norm(self, mode="elements"):
    if mode == "elements":
        return self / abs(self).max()
    elif mode == "full":
        return self / max(list(abs(self).max()))

@pmethod
def whiten(self):
    return Points(whiten(self.data))

@pstaticmethod
def angles(p1, p2):
    return (p1.cross(p2) / (abs(p1) * abs(p2))).asines()

@pstaticmethod
def angle(p1, p2):
    return abs(Points.angles(p1, p2))

@pstaticmethod
def X(value):
    return Points(np.array([value, np.zeros(value.shape), np.zeros(value.shape)]).T)

@pstaticmethod
def Y(value):
    return Points(np.array([np.zeros(value.shape), value, np.zeros(value.shape)]).T)

@pstaticmethod
def Z(value):
    return Points(np.array([np.zeros(value.shape), np.zeros(value.shape), value]).T)