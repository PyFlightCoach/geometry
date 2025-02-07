import numpy.typing as npt
import numpy as np



def get_index(arr: npt.NDArray, value: float):
    """given a value, find the index location in the aray,
    if no exact match, linearly interpolate in the index
    assumes arr is monotonic increasing
    raise value error if no match"""
    
    res = np.argwhere(arr==value)
    if len(res):
        return res[0][0]

    if value > arr[-1] or value < arr[0]:
        raise ValueError(f"Time {value} is out of bounds")
    
    i0=np.nonzero(arr<=value)[0][-1]
    i1=i0 + 1
    t0 = arr[i0]
    t1 = arr[i1]
    
    return i0 + (value - t0) / (t1 - t0)


def get_value(arr: npt.NDArray, index:float):
    """given an index, find the value in the array
    linearly interpolate if no exact match, 
    assumes arr is monotonic increasing"""
    if index < 0 or index > len(arr)-1:
        raise ValueError(f"Index {index} is out of bounds")
    
    frac = index % 1
    if frac == 0:
        return arr[int(index)]
    
    i0 = np.trunc(index)
    i1 = i0 + 1

    v0 = arr[int(i0)]
    v1 = arr[int(i1)]
    return v0 + (v1 - v0) * frac
