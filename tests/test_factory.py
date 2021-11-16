from geometry.factory import geom_factory, geoms_factory
import pytest
import numpy as np


def test_make_thing():
    Thing = geom_factory("Test", ["test1", "test2"])
    
    assert Thing.names[0] == "test1"

    thing = Thing(1, 2)

    np.testing.assert_array_equal(thing.test1, 1)
    np.testing.assert_array_equal(thing.test2, 2)

    dthing = thing.to_dict()

    assert dthing["test1"] == 1

def test_make_things():
    Things = geoms_factory("Tests", ["test1", "test2"], geom_factory("Test", ["test1", "test2"]))
    assert Things.names[0] == "test1"
    data = np.random.random((2,20))
    thing = Things(data)
    np.testing.assert_array_equal(thing.test1, data[:,0])

    
    