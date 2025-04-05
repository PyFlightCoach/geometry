from numbers import Number
from typing import Callable, Literal
import numpy.typing as npt
import numpy as np
import pandas as pd


def handle_slice(fun: Callable[[npt.NDArray, Number], Number]):
    def inner(
        arr: npt.NDArray, value: slice | Number | npt.ArrayLike | None, *args, **kwargs
    ) -> slice | Number | None:
        if isinstance(value, slice):
            start = (
                fun(arr, value.start, *args, **kwargs)
                if value.start is not None
                else None
            )
            stop = (
                fun(arr, value.stop, *args, **kwargs)
                if value.stop is not None
                else None
            )
            step = None  # TODO not sure how to handle this
            return slice(start, stop, step)
        elif pd.api.types.is_list_like(value):
            return [fun(arr, val, *args, **kwargs) for val in value]
        else:
            return None if value is None else fun(arr, value, *args, **kwargs)

    return inner


@handle_slice
def get_index(
    arr: npt.NDArray,
    value: Number,
    missing: float | Literal["throw"] = "throw",
    direction: Literal["forward", "backward"] = "forward",
):
    """given a value, find the index of the first location in the aray,
    if no exact match, linearly interpolate in the index
    assumes arr is monotonic increasing
    raise value error outside of bounds and missing == "throw", else return missing"""
    increasing = np.sign(np.diff(arr).mean())
    res = np.argwhere(arr == value)
    if len(res):
        return res[0 if direction == "forward" else -1, 0]
        # res[:,0]
    if value > arr.max() or value < arr.min():
        if missing == "throw":
            raise ValueError(f"Time {value} is out of bounds")
        else:
            return missing

    i0 = np.nonzero(arr <= value if increasing > 0 else arr >= value)[0][-1]
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


def apply_index_slice(index: npt.NDArray, value: slice | Number | npt.ArrayLike | None):
    if isinstance(value, slice):
        middle = pd.Index(index)[
            int(np.ceil(value.start)) if value.start is not None else None : int(
                np.ceil(value.stop)
            )
            if value.stop is not None
            else None
        ]
        if value.start is not None and middle[0] != value.start and value.start > index[0]:
            middle = np.concatenate([[get_value(index, value.start)], middle])
        if value.stop is not None and middle[-1] != value.stop and value.stop < index[-1]:
            middle = np.concatenate([middle, [get_value(index, value.stop)]])
        return middle
    else:
        return index[value]
