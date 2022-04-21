from geometry.mass import Mass
from geometry import PX
import numpy as np




def test_point():
    p = Mass.point(1)
    np.testing.assert_array_equal(
        p.data, 
        np.concatenate([[1], np.zeros(9)]).reshape(1,10)
    )

def test_sphere():
    s = Mass.sphere(1, 1)

    np.testing.assert_array_equal(
        s.data, 
        np.concatenate([[1], 2/5 * np.identity(3).reshape(9)]).reshape(1,10)
    )


def test_cuboid():
    c = Mass.cuboid(1, 1, 2, 3)

    np.testing.assert_array_almost_equal(
        c.data, 
        np.concatenate([[1], ([
            (2**2 + 3**2) / 12,
            (1**2 + 3**2) / 12,
            (1**2 + 2**2) / 12
        ] * np.identity(3)).reshape(9)]).reshape(1,10)
    )

def test_matrix():
    m = Mass.sphere(1, 1)
    np.testing.assert_array_almost_equal(
        m.matrix()[0],
        2/5 * np.identity(3)
    )

    mf = Mass.full(m, 10)
    np.testing.assert_array_almost_equal(
        mf.matrix(),
        np.tile(2/5 * np.identity(3), (10,1,1))
    )

def test_offset():
    m = Mass.point(1).offset(PX(1))

    assert m.m[0] == 1
    assert m.xx[0] == 0
    assert m.yy[0] == 1
    assert m.zz[0] == 1

    