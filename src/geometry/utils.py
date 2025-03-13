from numbers import Number
from typing import Callable, Literal
import numpy.typing as npt
import numpy as np


def handle_slice(fun: Callable[[npt.NDArray, Number], Number]):
    def inner(arr: npt.NDArray, value: slice | Number | None) -> slice | Number | None:
        if isinstance(value, slice):
            start = fun(arr, value.start) if value.start is not None else None
            stop = fun(arr, value.stop) if value.stop is not None else None
            step = None  # TODO not sure how to handle this
            return slice(start, stop, step)
        else:
            return None if value is None else fun(arr, value)

    return inner


@handle_slice
def get_index(arr: npt.NDArray, value: Number, missing: int | Literal["raise"] = "throw"):
    """given a value, find the index location in the aray,
    if no exact match, linearly interpolate in the index
    assumes arr is monotonic increasing
    raise value error if no match and missing == "throw", else return missing"""
    direction = np.sign(np.diff(arr).mean())
    res = np.argwhere(arr == value)
    if len(res):
        return res[0, 0]
        # res[:,0]
    if value > arr.max() or value < arr.min():
        if missing=="throw":
            raise ValueError(f"Time {value} is out of bounds")
        else:
            return missing

    i0 = np.nonzero(arr <= value if direction > 0 else arr >= value)[0][-1]
    i1 = i0 + 1
    t0 = arr[i0]
    t1 = arr[i1]

    return i0 + (value - t0) / (t1 - t0)


@handle_slice
def get_value(arr: npt.NDArray, index: Number):
    """given an index, find the value in the array
    linearly interpolate if no exact match,
    assumes arr is monotonic increasing"""
    if index > len(arr) - 1:
        raise ValueError(f"Index {index} is out of bounds")
    elif index < 0:
        index = len(arr) + index
    frac = index % 1
    if frac == 0:
        return arr[int(index)]

    i0 = np.trunc(index)
    i1 = i0 + 1

    v0 = arr[int(i0)]
    v1 = arr[int(i1)]
    return v0 + (v1 - v0) * frac
