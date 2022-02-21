import numpy as np
from geometry.points import Point, Points


class Points:

    def __abs__(self) -> np.ndarray:
        ...

    @property
    def x(self) -> np.ndarray:
        ...

    @property
    def y(self) -> np.ndarray:
        ...

    @property
    def z(self) -> np.ndarray:
        ...

    def scale(self, value: float) -> Points:
        """scale the point by value

        Args:
            value (float): _description_

        Returns:
            Points: _description_
        """
        ...

    def unit(self) -> Points:
        ...