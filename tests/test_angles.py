import numpy as np
import geometry as g


def test_unwind():
    assert g.angles.unwind(0) == 0
    assert g.angles.unwind(2*np.pi) == 0
    assert g.angles.unwind(3*np.pi) == -np.pi
    assert g.angles.unwind(-2*np.pi) == 0
    assert g.angles.unwind(-3*np.pi) == np.pi
    assert g.angles.unwind(3*np.pi, np.pi) == np.pi
    assert g.angles.unwind(-np.pi, np.pi) == np.pi

def test_unwind_array():
    np.testing.assert_array_almost_equal(
        g.angles.unwind(np.zeros(10)),
        np.zeros(10)
    )