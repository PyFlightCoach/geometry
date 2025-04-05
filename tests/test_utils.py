from pytest import raises, mark
from geometry.utils import get_index, get_value, apply_index_slice
import numpy as np

def test_get_index():
    arr = np.arange(10)
    assert get_index(arr, 5) == 5
    assert get_index(arr, 5.5) == 5.5
    assert get_index(arr, 0) == 0
    assert get_index(arr, 9) == 9
    with raises(ValueError):
        get_index(arr, -1)
    with raises(ValueError):
        get_index(arr, 10)


def test_get_index_repeat():
    arr = np.array([0, 1, 1, 1, 2, 3, 4])
    assert get_index(arr, 3) == 5
    assert get_index(arr, 1) == 1
    assert get_index(arr, 1, direction="backward") == 3
    assert get_index(arr, 0.5) == 0.5
    assert get_index(arr, 1.5) == 3.5


def test_get_value():   
    arr = np.arange(10)
    assert get_value(arr, 5) == 5
    assert get_value(arr, 5.5) == 5.5
    assert get_value(arr, 0) == 0
    assert get_value(arr, 9) == 9
    with raises(ValueError):
        get_value(arr, 10)


def test_apply_index_slice():
    assert np.all(apply_index_slice(np.arange(10), slice(1, 5)) == np.array([1, 2, 3, 4, 5]))
    assert np.all(apply_index_slice(np.arange(10), slice(0.5, 1.5)) == np.array([0.5, 1, 1.5]))
    assert np.all(apply_index_slice(np.arange(10), slice(0.5, 3.5)) == np.array([0.5, 1,  2, 3, 3.5]))
    assert np.all(apply_index_slice(np.arange(1), slice(0, 5)) == np.array([0]))
    