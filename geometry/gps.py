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
from geometry.point import Point
from geometry.points import Points
from typing import List, Union
import numpy as np
import pandas as pd

def safecos(angledeg: Union[float, np.ndarray]):
    if isinstance(angledeg, float):
        return max(np.cos(np.radians(angledeg)), 0.01)
    elif isinstance(angledeg, np.ndarray):
        return np.maximum(np.cos(np.radians(angledeg)), np.full(len(angledeg), 0.01))
    

class GPSPosition(object):
    # was 6378137, extra precision removed to match ardupilot
    approx_earth_radius = 6378100
    LOCATION_SCALING_FACTOR = math.radians(approx_earth_radius)

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
        self._longitude_scale = safecos(self.latitude)

    def __getattr__(self, name):
        if name in ["lat", "la"]:
            return self.latitude
        if name in ["long", "lon", "lo"]:
            return self.longitude

    def to_list(self):
        return [self.latitude, self.longitude]

    def to_tuple(self):
        return (self.latitude, self.longitude)

    def to_dict(self):
        return dict(latitude=self.latitude, longitude=self.longitude)

    def __str__(self):
        return 'lat: ' + str(self.latitude) + ', long: ' + str(self.longitude)

    def offset(self, pin: Union[Point, Points]):
        latb = self.latitude - pin.x / self.LOCATION_SCALING_FACTOR

        return GPSPosition(
            latb,
            self.longitude + pin.y / (self.LOCATION_SCALING_FACTOR * safecos(latb))
        )

    def _to_xy(self):
        lat = self.latitude * math.pi / 180
        lon = self.longitude * math.pi / 180

        return [
            GPSPosition.approx_earth_radius * math.cos(lat) * math.cos(lon),
            GPSPosition.approx_earth_radius * math.cos(lat) * math.sin(lon)
        ]

    def __sub__(self, other) -> Point:
        """Gives the ned vector from other to self

        Args:
            other (GPSPostion): [description]

        Returns:
            Point: vector from other to self
        """
        return Point(
            (other.latitude - self.latitude) * GPSPosition.LOCATION_SCALING_FACTOR,
            -(other.longitude - self.longitude) * GPSPosition.LOCATION_SCALING_FACTOR * self._longitude_scale,
            0
        )

    def __eq__(self, other) -> bool:
        return self.lat == other.lat and self.long == other.long




'''
// scaling factor from 1e-7 degrees to meters at equator
// == 1.0e-7 * DEG_TO_RAD * RADIUS_OF_EARTH
static constexpr float LOCATION_SCALING_FACTOR = 0.011131884502145034f;
// inverse of LOCATION_SCALING_FACTOR
static constexpr float LOCATION_SCALING_FACTOR_INV = 89.83204953368922f;

Vector3f Location::get_distance_NED(const Location &loc2) const
{
    return Vector3f((loc2.lat - lat) * LOCATION_SCALING_FACTOR,
                    (loc2.lng - lng) * LOCATION_SCALING_FACTOR * longitude_scale(),
                    (alt - loc2.alt) * 0.01f);
}

float Location::longitude_scale() const
{
    float scale = cosf(lat * (1.0e-7f * DEG_TO_RAD));
    return MAX(scale, 0.01f);
}
'''


if __name__ == "__main__":
    home = GPSPosition(51.459387, -2.791393)

    new = GPSPosition(51.458876, -2.789092)
    coord = home - new
    print(coord.x, coord.y)
