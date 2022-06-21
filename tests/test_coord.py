from pytest import approx, mark
from geometry import Coord, Point, Quaternion, P0, PX, PY, PZ
import numpy as np



def test_axes():
    coord = Coord(np.ones((20, 12)))
    assert coord.origin == Point(np.ones((20,3)))

def test_from_axes():
    coord = Coord.from_axes(P0(2), PX(1,2), PY(1,2), PZ(1,2))
    assert coord.data[:,:3] == P0(2)


def test_rotation_matrix():
    np.testing.assert_array_equal(
        Coord.from_nothing(10).rotation_matrix(),
        np.tile(np.identity(3).reshape(1,3,3), (10,1,1))
    )


def test_rotate():
    q = Quaternion.from_euler(Point(0, 0, np.pi/2))

    c = Coord.from_nothing(10)

    rc = c.rotate(q)
    assert rc.origin == P0()
    np.testing.assert_almost_equal(rc.x_axis.data, PY(1,10).data)
    np.testing.assert_almost_equal(rc.y_axis.data, PX(-1,10).data)
    np.testing.assert_almost_equal(rc.z_axis.data, PZ(1,10).data)
