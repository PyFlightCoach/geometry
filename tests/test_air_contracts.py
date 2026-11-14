"""Air columns retain SI pressure/temperature and the ideal-gas density."""

import numpy as np
import pytest

from geometry import Air
from geometry.air import get_rho


def test_scalar_pressure_temperature_produce_the_known_density():
    # p = rho R T, with R_air = 287.058 J/(kg K). Choose rho=2 kg/m^3.
    pressure = 2 * 287.058 * 300
    air = Air.from_pt(pressure, 300)
    np.testing.assert_allclose(air.data, [[pressure, 300, 2]], rtol=0, atol=1e-12)
    assert get_rho(pressure, 300) == pytest.approx(2, abs=1e-15)


def test_vector_pressure_temperature_preserve_independent_rows_and_inputs():
    temperature = np.array([250., 300., 320.])
    expected_density = np.array([.8, 1.2, 2.])
    pressure = expected_density * 287.058 * temperature
    before = pressure.copy(), temperature.copy()
    air = Air.from_pt(pressure, temperature)
    np.testing.assert_array_equal(air.P, pressure)
    np.testing.assert_array_equal(air.T, temperature)
    np.testing.assert_allclose(air.rho, expected_density, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(pressure, before[0])
    np.testing.assert_array_equal(temperature, before[1])


@pytest.mark.parametrize("count", [0, 1, 4])
def test_standard_sea_level_repeats_the_physical_state(count):
    air = Air.iso_sea_level(count)
    expected = np.tile([101325., 288.15, 101325 / (287.058 * 288.15)], (count, 1))
    assert air.data.shape == (count, 3)
    np.testing.assert_allclose(air.data, expected, rtol=0, atol=1e-12)
