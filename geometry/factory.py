import numpy as np
import pandas as pd


def gfactory(name, keys):
    Plural = geoms_factory(name + "s", keys)
    Single = geom_factory(Plural)

    Plural.__getitem__ = lambda self, i: Single(self.data[i, :])

    return Plural, Single


def geoms_factory(name, names):
    assert len(names) > 1
    def __init__(self, data):
        assert data.shape[1] == len(names)
        self.data = data

    def __getattr__(self, name):
        return self.data[:, self.names.index(name)]

    def to_pandas(self, prefix='', suffix='', columns=names, index=None):
        return pd.DataFrame(
            self.data, 
            columns=[prefix + col + suffix for col in columns],
            index=index
        ) 

    def __dir__(self):
        return self.super().__dir__() + names

    def count(self):
        return self.data.shape[0]


    Obj =  type(name, (object,),
        {   
            "names": names,
            "__init__": __init__,
            "__getattr__": __getattr__,
            "__dir__": __dir__,
            "__repr__": lambda self: f"{name}({self.data})",
            "__str__": lambda self: f"{self.data}",
            "__eq__": lambda self, other: np.all(self.data == other.data),
            "to_pandas": to_pandas,
            "count": property(count)
        })
    
    Obj.from_pandas = staticmethod(lambda df: Obj(np.array(df)))
    
    return Obj

def geom_factory(Plural):
    name = Plural.__name__[:-1]
    Obj =  type(name, (Plural,),
        {
            "names": Plural.names,
            "__getattr__": lambda self, key: self.data[0, Plural.names.index(key)],
            "__repr__": lambda self: f"{name}({list(self.data[0,:])})",
            "__str__": lambda self: f"{list(self.data[0,:])}",
            "to_list": lambda self: list(self.data[0,:]),
        })

    def __init__(self,*args,**kwargs):
        if len(args) == 1:
            data = np.array(args[0])
        if "data" in kwargs.keys():
            data = np.array(kwargs["data"])
        
        if len(args) == len(Plural.names):
            data = np.array(args).reshape((len(Plural.names),1))

        try:
            data
        except NameError:
            assert np.all([key in kwargs.keys() for key in Plural.names])
            data = np.zeros((1, len(Plural.names)))

            for i, key in enumerate(Plural.names):
                if key in kwargs.keys():
                    data[0,i] = kwargs[key]

        data = data.reshape((1, len(Plural.names)))
        assert data.shape[1] == len(Plural.names)
        assert data.shape[0] == 1
        
        super(Obj, self).__init__(data)

    Obj.__init__ = __init__

    Obj.full = lambda self, count: Plural(np.tile(self.data, (count, 1)))

    return Obj