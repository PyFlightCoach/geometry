import numpy as np
import numpy.typing as npt

def unwind(angles: npt.NDArray, center: float = 0.0):
    """given an angle, unwind it to the range [-pi, pi] around a center point."""
    turns = np.round((angles - center) / (2 * np.pi)) 
    return angles - turns * 2 * np.pi


def difference(a, b): 
    d1 = a - b
    d2 = d1 - 2 * np.pi
    bd=np.abs(d2) < np.abs(d1)
    d1[bd] = d2[bd]
    d3 = d1 + 2 * np.pi
    bd=np.abs(d3) < np.abs(d1)
    d1[bd] = d3[bd]

    return d1