from pytest import raises
from geometry.utils import get_index, get_value
import numpy as np

def test_get_index():
    arr = np.arange(10)
    assert get_index(arr, 5)[0] == 5
    assert get_index(arr, 5.5)[0] == 5.5
    assert get_index(arr, 0)[0] == 0
    assert get_index(arr, 9)[0] == 9
    with raises(ValueError):
        get_index(arr, -1)[0]
    with raises(ValueError):
        get_index(arr, 10)[0]


def test_get_index_repeat():
    arr = np.array([0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9])
    assert get_index(arr, 3)[0] == 3
    assert get_index(arr, 3)[-1] == 4

def test_get_value():   
    arr = np.arange(10)
    assert get_value(arr, 5) == 5
    assert get_value(arr, 5.5) == 5.5
    assert get_value(arr, 0) == 0
    assert get_value(arr, 9) == 9
    with raises(ValueError):
        get_value(arr, -1)
    with raises(ValueError):
        get_value(arr, 10)