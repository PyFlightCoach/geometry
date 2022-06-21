from pytest import mark, approx
from geometry import Coord, Transformation, Point, Quaternion, PX, PY, PZ, P0
import numpy as np



def test_from_coords():
    c1 = Coord.from_xy(P0(), PX(), PY())
    c2 = Coord.from_xy(P0(), PY(), PZ())
    trans_to = Transformation.from_coords(c1,c2)
    trans_from = Transformation.from_coords(c2,c1)
    
    ps = Point(np.random.random((100, 3)))
    
    np.testing.assert_array_almost_equal(
        ps.data,
        trans_from.translate(trans_to.translate(ps)).data
    )

    qs = Quaternion.from_euler(ps)
    np.testing.assert_array_almost_equal(
        qs,
        trans_from.rotate(trans_to.rotate(qs))
    )

   

def test_translate():
    ca = Coord.from_nothing()
    cb = Coord.from_nothing().translate(Point(1, 0, 0))
    transform = Transformation.from_coords(ca, cb)
    assert transform.translate(Point(0, 0, 0)) == Point(1, 0, 0)


def _test_rotate(c1, c2, p1, p2):
    transform = Transformation.from_coords(c1, c2)
    p1b = transform.rotate(p1)
    np.testing.assert_almost_equal(p1b.data, p2.data, err_msg=f'{p1b} != {p2}')
    

def test_rotate():
    _test_rotate(
        Coord.from_nothing(),
        Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)),
        Point(1, 1, 0),
        Point(1, -1, 0)
    )

    _test_rotate(
        Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)),
        Coord.from_zx(Point(0, 0, 0), Point(0, 0, 1), Point(0, -1, 0)),
        Point(1, 1, 0),
        Point(-1, -1, 0)
    )



def test_to_matrix():
    np.testing.assert_array_equal(
        Transformation().to_matrix(), 
        np.identity(4).reshape(1,4,4)
    )

    
