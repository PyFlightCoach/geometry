from geometry.base import Base

R = 287.058
GAMMA = 1.4


def get_rho(pressure, temperature):
    return pressure / (R * temperature)


class Air(Base):
    cols = ["P", "T", "rho"]

    @staticmethod
    def iso_sea_level(length: int):
        return Air(101325, 288.15, get_rho(101325, 288.15)).tile(length)

    @staticmethod
    def from_pt(pressure, temperature):
        return Air(pressure, temperature, get_rho(pressure, temperature))
