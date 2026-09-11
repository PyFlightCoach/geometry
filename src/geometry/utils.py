from collections.abc import Callable
from numbers import Number
from typing import Literal

import numpy as np
import numpy.typing as npt
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
    missing: float | Literal["throw"] | None = "throw",
    direction: Literal["forward", "backward"] = "forward",
    increasing: bool | None = None,
):
    """given a value, find the index of the first location in the aray,
    if no exact match, linearly interpolate in the index
    assumes arr is monotonic increasing
    raise value error outside of bounds and missing == "throw", else return missing
    increasing, is the array going up or down, if not given it will be inferred from the data
    """

    #    index = np.searchsorted(arr, value, "left" )

    increasing = np.sign(np.diff(arr).mean()) if increasing is None else increasing
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
        if (
            value.start is not None
            and value.stop is not None
            and value.start >= value.stop
        ):
            return np.array([], dtype=index.dtype)
        middle = pd.Index(index)[
            (int(np.ceil(value.start)) if value.start is not None else None) : (
                int(np.ceil(value.stop)) if value.stop is not None else None
            )
        ].values

        if (
            value.start is not None
            and (len(middle) == 0 or middle[0] != value.start)
            and value.start > index[0]
        ):
            middle = np.concatenate([[get_value(index, value.start)], middle])
        if (
            value.stop is not None
            and (len(middle) == 0 or middle[-1] != value.stop)
            and value.stop < index[-1]
        ):
            middle = np.concatenate([middle, [get_value(index, value.stop)]])
        return middle
    else:
        return index[value]


def round_slice(sli: npt.ArrayLike | slice | Number) -> npt.ArrayLike | slice | Number:
    if isinstance(sli, slice):
        return slice(
            int(np.floor(sli.start)) if sli.start is not None else None,
            int(np.ceil(sli.stop)) if sli.stop is not None else None,
        )
    elif isinstance(sli, Number):
        return int(sli)
    elif pd.api.types.is_list_like(sli):
        return np.array([int(np.floor(s)) for s in sli[:-1]] + [int(np.ceil(sli[-1]))])
    else:
        raise ValueError(f"Cannot expand {sli}")


def inclusive_slice(
    arr: npt.ArrayLike, sli: slice | Number | npt.ArrayLike | None
) -> npt.ArrayLike:
    """slice an array but include the right boundary"""

    if isinstance(sli, slice):
        oarr = arr[sli]
        return (
            np.concatenate([oarr, [oarr[-1] + 1]])
            if sli.stop is not None and sli.stop <= (len(arr) - 1)
            else oarr
        )
    elif isinstance(sli, Number):
        return arr[int(sli)]
    elif pd.api.types.is_list_like(sli):
        return arr[sli]
    elif sli is None:
        return arr
    else:
        raise ValueError(f"Cannot expand {sli}")


def make_smoothing_spline(data: npt.ArrayLike, index: npt.ArrayLike = None, auto_s: bool = True, auto_s_cutoff_freq = 10, min_s: float=1e-4, **kwargs):
    from scipy.interpolate import make_splrep
    from scipy.signal import butter, sosfiltfilt
    
    index = np.arange(len(data)) if index is None else index

    if auto_s:
        #if the weights are equal to 1, s should be chosen based on the nuimber of points and the noise variance
        #Dierckx, P. (1981). An algorithm for cubic spline fitting with convexity constraints. Computing, 26(4), 327–334.
        
        # using a high pass filter to isolate the noise variance:
        #Schulze, H. G., et al. (2011). Denoising of spectra with no user input: a spline‐smoothing approach. Journal of Raman Spectroscopy, 42(8), 1630-1638.
        
        sos = butter(
            N=4,
            Wn=auto_s_cutoff_freq,
            btype="highpass",
            fs=1 / np.median(np.diff(index)),
            output="sos",
        )
    
        _noise = sosfiltfilt(sos, data)
        _trim = int(len(data) * 0.05)
        _noise_variance = np.var(_noise[1+_trim : -2-_trim])
        
        kwargs['s'] = _noise_variance * len(index)
    
    kwargs['s'] = max(kwargs.get("s", min_s), min_s)

    w = np.ones(len(index))
    trim_len = int(len(index) * 0.05)
    w.put(np.arange(trim_len), 0.5)  
    w.put(np.arange(len(index) - trim_len, len(index)), 0.5)

    return make_splrep(index, data, w=w, **kwargs)
