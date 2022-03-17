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

    def _dprep(self, other):        
        arr = other.data if isinstance(other, Base) else other
        l , w = len(self), len(self.cols)
        if isinstance(arr, np.ndarray):
            if arr.shape == (l,w):
                return arr
            elif arr.shape == (l, 1) or arr.shape == (l,):
                return np.tile(arr, (w,1)).T
            elif arr.shape == (1, w) or arr.shape == (w,):
                return np.tile(arr, (l,1))
            elif arr.shape == (1,):
                return np.full((l,w), arr[0])
            elif l==1:
                return arr
            else:
                raise ValueError(f"array shape {arr.shape} not handled")
        elif isinstance(arr, float) or isinstance(arr, int):
            return np.full((l,w), arr)
        elif isinstance(other, Base):
            assert len(other) == 1 or len(other) == l
            return other.data
        else:
            raise ValueError(f"unhandled other datatype ({other.__class__.name})")

    def count(self):
        return len(self)

    def __len__(self):
        return self.data.shape[0]


    def __eq__(self, other):
        return np.all(self.data==self._dprep(other))

    def __add__(self, other):
        return self.__class__(self.data + self._dprep(other))
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self.__class__(self.data - self._dprep(other))
    
    def __rsub__(self, other):
        return self.__class__(self._dprep(other) - self.data)

    def __mul__(self, other):
        return self.__class__(self.data * self._dprep(other))

    def __rmul__(self, other):
        return self.__class__(self._dprep(other) * self.data)

    def __rtruediv__(self, other):
        return self.__class__(self._dprep(other) / self.data)

    def __truediv__(self, other):
        return self.__class__(self.data / self._dprep(other))

    def __str__(self):
        return str(pd.DataFrame(self.data, columns=self.__class__.cols))

    def __abs__(self):
        return np.linalg.norm(self.data, axis=1)

    def __neg__(self):
        return -1 * self

    def dot(self, other):
        return np.einsum('ij,ij->i', self.data, self._dprep(other))


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
