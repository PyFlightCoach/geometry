from geometry.factory import geom_factory, geoms_factory, gfactory
import pytest
import numpy as np



@pytest.fixture(scope="session")
def Things():
    return geoms_factory("Things", ["test1", "test2"])
    

def test_make_things(Things):
    assert Things.names[0] == "test1"
    assert Things.__name__ == "Things"
    data = np.random.random((20,2))
    thing = Things(data)
    np.testing.assert_array_equal(thing.test1, data[:,0])



@pytest.fixture(scope="session")
def Thing(Things):
    return geom_factory(Things)


def test_make_thing_from_data(Thing):
       
    thing = Thing([233, 235])
    assert thing.test1 == 233
    assert thing.test2 == 235


def test_make_thing_from_names(Thing):
    thing = Thing(23, 45)

    assert thing.test1 == 23
    assert thing.test2 == 45


def test_make_thing_kwargs(Thing):
    th = Thing(test1=1,test2=2)
    assert th.test1 ==1

def test_full():
    Ths, Th = gfactory("Th", list("abc"))
    th = Th(1,2,3)
    ths = th.full(100)
    assert ths.count == 100
    np.testing.assert_array_equal(ths.a, np.full(100, 1))


def test_get_item():
    Ths, Th = gfactory("Th", list("abc"))

    ths = Ths(np.full((10, 3), [1,2,3]))

    th = ths[5]
    assert np.all(th.to_list() == [1,2,3])

    assert np.all(th[0].to_list() == [1,2,3])

def test_eq():
    Ths, Th = gfactory("Th", list("abc"))

    assert Th(1,2,3) == Th([1,2,3])

