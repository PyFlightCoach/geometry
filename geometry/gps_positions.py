import numpy as np
import pandas as pd

from geometry.point import Point
from geometry.points import Points
from geometry.gps import safecos, GPSPosition

from typing import Union


class GPSPositions(object):
    def __init__(self, data: np.array):
        self.data = data
        self._longitude_scale = safecos(self.latitude)
        assert data.shape[1] == 2

    @property
    def latitude(self):
        return self.data[:, 0]

    @property
    def longitude(self):
        return self.data[:, 1]

    def __getattr__(self, name):
        if name in ["lat", "la"]:
            return self.latitude
        if name in ["long", "lon", "lo"]:
            return self.longitude

    @property
    def count(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return GPSPosition(*list(self.data[index, :]))

    @staticmethod
    def from_pandas(df):
        return GPSPositions(np.array(df))

    def to_pandas(self, prefix='', suffix='', columns=['latitude', 'longitude']):
        return pd.DataFrame(self.data, columns=[prefix + col + suffix for col in columns])

    @staticmethod
    def full(gps: GPSPosition, count: int):
        return GPSPositions(np.tile(np.array(gps.to_list()), (count, 1)))

    def __sub__(self, other) -> Points:
        if isinstance(other, GPSPosition):
            others = GPSPositions.full(other, self.count)
        else:
            others = other
        assert self.count == others.count
        return Points(np.column_stack([
            -(others.latitude - self.latitude) * GPSPosition.LOCATION_SCALING_FACTOR,
             -(others.longitude - self.longitude) * GPSPosition.LOCATION_SCALING_FACTOR * self._longitude_scale,
             np.zeros(self.count)
        ]))

    def offset(self, pin: Union[Point, Points]):
        if isinstance(pin, Point):
            pin = Points.full(pin, self.count)
        assert pin.count == self.count
        latb = self.latitude + pin.x / GPSPosition.LOCATION_SCALING_FACTOR

        return GPSPositions(
            np.column_stack([
                latb,
                self.longitude + pin.y / (GPSPosition.LOCATION_SCALING_FACTOR * safecos(latb))
            ])
        )
