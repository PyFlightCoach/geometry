"""This module contains helper functions for unit testing geometry objects"""
import numpy as np
from geometry import Base


def assert_equal(a: Base,b: Base, *args, **kwargs):
    np.testing.assert_array_equal(a.data,b.data, *args, **kwargs)

def assert_almost_equal(a:Base,b:Base, *args, **kwargs):
    np.testing.assert_array_almost_equal(a.data,b.data, *args, **kwargs)

