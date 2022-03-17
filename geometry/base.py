from typing import Type, List
import numpy as np
import pandas as pd


class Base:
    __array_priority__ = 15.0   # this is a quirk of numpy so the __r*__ methods here take priority
    cols=[]
    from_np_base = []
    from_np = []
    def __init__(self, *args, **kwargs):
        if len(kwargs) > 0:
            if len(args) > 0:
                raise TypeError("Cannot accept args and kwargs at the same time")
            if all([c in kwargs for c in self.__class__.cols]):
                args = [kwargs[c] for c in self.__class__.cols]
            elif "data" in kwargs:
                args = [kwargs["data"]]
            else:
                raise TypeError("unknown kwargs passed")

        if len(args)==1: 
            if isinstance(args[0], np.ndarray): #data was passed directly
                self.data = self.__class__._clean_data(args[0])

            elif all([isinstance(a, self.__class__) for a in args[0]]):
                #a list of self.__class__ is passed, concatenate into one
                self.data = self.__class__._clean_data(np.concatenate([a.data for a in args[0]]))
            
            elif isinstance(args[0], pd.DataFrame):
                _cols = []
                for col in self.__class__.cols:
                    for dfcol in args[0].columns:
                        if col in dfcol:
                            _cols.append(dfcol)
                self.data = self.__class__._clean_data(np.array(args[0][_cols]))

        elif len(args) == len(self.__class__.cols):
            #three args passed, each represents a col
            self.data = self.__class__._clean_data(np.array(args).T)
        else:
            raise TypeError(f"Empty {self.__class__.__name__} not allowed")

    @classmethod
    def _clean_data(cls, data) -> np.ndarray:
        assert isinstance(data, np.ndarray)
        if data.dtype == 'O': 
            raise ValueError('data must have homogeneous shape')
        if len(data.shape) == 1:
            data = data.reshape(1, len(data))
        
        assert data.shape[1] == len(cls.cols)
        return data

    
    def __getattr__(self, name):
        if name in self.__class__.cols:
            return self.data[:,self.__class__.cols.index(name)]
        elif name in self.__class__.from_np + self.__class__.from_np_base:
            return self.__class__(getattr(np, name)(self.data))
        raise AttributeError(f"Cannot get attribute {name}")

    def __dir__(self):
        return self.__class__.cols

    def __getitem__(self, sli):
        return self.__class__(self.data[sli,:])

    @staticmethod
    def _data(other):
        return other.data if isinstance(other, Base) else other

    def count(self):
        return len(self)

    def __len__(self):
        return self.data.shape[0]

    def __eq__(self, other):
        return np.all(self.data==Base._data(other))

    def __add__(self, other):
        return self.__class__(self.data + Base._data(other))
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self.__class__(self.data - Base._data(other))
    
    def __rsub__(self, other):
        return self.__class__(Base._data(other) - self.data)

    def __mul__(self, other):
        return self.__class__(self.data * Base._data(other))

    def __rmul__(self, other):
        return self.__class__(Base._data(other) * self.data)

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(other.data / self.data)
        elif isinstance(other, np.ndarray):
            if other.shape == (len(self), 1):
                return np.tile(other, (4,1)).T / self
        return self.__class__(Base._data(other) / self.data)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.data / other.data)
        elif isinstance(other, np.ndarray):
            if other.shape == (len(self),):
                return self / np.tile(other, (4,1)).T    
        return self.__class__(self.data / Base._data(other))



    def __str__(self):
        return str(pd.DataFrame(self.data, columns=self.__class__.cols))

    def __abs__(self):
        return np.linalg.norm(self.data, axis=1)

    def __neg__(self):
        return -1 * self

    def dot(self, other):
        return np.einsum('ij,ij->i', self.data, Base._data(other))


    def diff(self, dt:np.array):
        assert len(dt) == len(self)
        return self.__class__(
            np.gradient(self.data,axis=0) \
                 / \
                np.tile(dt, (len(self.__class__.cols),1)).T)

    def to_pandas(self, prefix='', suffix='', index=None):
        return pd.DataFrame(
            self.data, 
            columns=[prefix + col + suffix for col in self.__class__.cols],
            index=index
        )

    @classmethod
    def full(cls, val, count):
        return cls(np.tile(val.data, (count, 1)))
