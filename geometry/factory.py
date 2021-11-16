import numpy as np
import pandas as pd



def geoms_factory(name, names, Single):
    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        return self.data[:, self.names.index(name)]

    def to_pandas(self, prefix='', suffix='', columns=names, index=None):
        return pd.DataFrame(
            self.data, 
            columns=[prefix + col + suffix for col in columns],
            index=index
        ) 
    
    def __getitem__(self, i):
        return Single(*self.data[i,:])

    Obj =  type(name, (object,),
        {   
            "names": names,
            "__init__": __init__,
            "__getattr__": __getattr__,
            "__getitem__": __getitem__,
            "to_pandas": to_pandas,
        })
    
    Obj.from_pandas = staticmethod(lambda df: Obj(np.array(df)))

    return Obj


def geom_factory(name, names):
    def __init__(self, *args, **kwargs):
        for name, value in zip(names, args):
            kwargs[name] = value
        assert set(kwargs.keys()) == set(names)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self, prefix=""):
        return {prefix + key: value for key, value in self.__dict__.items()}
        

    Obj =  type(name, (object,),
        {
            "names": names,
            "__init__": __init__,
            "to_dict": to_dict,
        })

    
    Obj.from_dict = staticmethod(lambda value: Obj(*list(value.values())))

    return Obj