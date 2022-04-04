"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import math
from geometry.base import Base
from geometry.point import Point
from typing import List, Union
import numpy as np
import pandas as pd


def safecos(angledeg: Union[float, int, np.ndarray]):
    if isinstance(angledeg, float) or isinstance(angledeg, int):
        return max(np.cos(np.radians(angledeg)), 0.01)
    elif isinstance(angledeg, np.ndarray):
        return np.maximum(np.cos(np.radians(angledeg)), np.full(len(angledeg), 0.01))


erad = 6378100
LOCFAC = math.radians(erad)


class GPS(Base):
    cols = ["lat", "long"]
    # was 6378137, extra precision removed to match ardupilot

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._longfac = safecos(self.lat)


    def offset(self, pin: Point):
        assert len(pin) == len(self)
        latb = self.lat - pin.x / LOCFAC

        return GPS(
            latb,
            self.long + pin.y / (LOCFAC * safecos(latb))
        )

    def __eq__(self, other) -> bool:
        return np.all(self.data == other.data)

    def __sub__(self, other) -> Point:
        assert isinstance(other, GPS)
        if len(other) == len(self):
            return Point(
                -(other.lat - self.lat) * LOCFAC,
                -(other.long - self.long) * LOCFAC * self._longfac,
                np.zeros(len(self))
            )
        elif len(other) == 1:
            return self - GPS.full(other, len(self))
        elif len(self) == 1:
            return GPS.full(self, len(self)) - other
        else:
            raise ValueError(f"incompatible lengths for sub ({len(self)}) - ({len(other)})")

    def offset(self, pin: Point):
        if len(pin) == 1 and len(self) > 1:
            pin = Point.full(pin, self.count)
        elif len(self) == 1 and len(pin) > 1:
            return self.full(len(pin)).offset(pin)
        
        if not len(pin) == len(self):
            raise ValueError(f"incompatible lengths for offset ({len(self)}) - ({len(pin)})")

        latb = self.lat + pin.x / LOCFAC
        return GPS(
            latb,
            self.long + pin.y / (LOCFAC * safecos(latb))
        )



'''
// scaling factor from 1e-7 degrees to meters at equator
// == 1.0e-7 * DEG_TO_RAD * RADIUS_OF_EARTH
static constexpr float LOCATION_SCALING_FACTOR = 0.011131884502145034f;
// inverse of LOCATION_SCALING_FACTOR
static constexpr float LOCATION_SCALING_FACTOR_INV = 89.83204953368922f;

Vector3f Location::get_distance_NED(const Location &loc2) const
{
    return Vector3f((loc2.lat - lat) * LOCATION_SCALING_FACTOR,
                    (loc2.lng - lng) * LOCATION_SCALING_FACTOR * long_scale(),
                    (alt - loc2.alt) * 0.01f);
}

float Location::long_scale() const
{
    float scale = cosf(lat * (1.0e-7f * DEG_TO_RAD));
    return MAX(scale, 0.01f);
}
'''


if __name__ == "__main__":
    home = GPS(51.459387, -2.791393)

    new = GPS(51.458876, -2.789092)
    coord = home - new
    print(coord.x, coord.y)
