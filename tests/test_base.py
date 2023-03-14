from geometry.base import Base
from pytest import mark, approx, raises
import numpy as np
import pandas as pd

a_b_c = ["a", "b", "c"]
class ABC(Base):
    cols=list("abc")


def test_init_nparray():
    abc = ABC(np.ones((5,3)))
    np.testing.assert_array_equal(abc.data, np.ones((5,3)))

    abc = ABC(np.ones(3))
    np.testing.assert_array_equal(abc.data, np.ones(3).reshape((1,3)))



def test__dprep():
    np.testing.assert_array_equal(
        ABC(np.ones((10,3)))._dprep(np.ones(10)), 
        np.ones((10,3))
    )

    np.testing.assert_array_equal(
        ABC(np.ones((10,3)))._dprep(np.ones((10,3))), 
        np.ones((10,3))
    )

    with raises(ValueError):
        ABC(np.ones((10,3)))._dprep(np.ones((5,3)))

    with raises(ValueError):
        np.testing.assert_array_equal(
            ABC(np.ones((10,3)))._dprep(np.ones(3)), 
            np.ones((10,3))
        )

    np.testing.assert_array_equal(
        ABC(np.ones((10,3)))._dprep(np.array([1])), 
        np.ones((10,3))
    )

    np.testing.assert_array_equal(
        ABC(np.ones((10,3)))._dprep(1), 
        np.ones((10,3))
    )


def test_dot():
    a = ABC(np.random.random((10, 3)))
    b = ABC(np.random.random((10, 3)))

    c = a.dot(b)

    c_check = np.array([np.dot(_a.data[0], _b.data[0]) for _a, _b in zip(a,b)])

    np.testing.assert_array_almost_equal(c, c_check)


def test_init_values():
    abc = ABC(1,2,3)
    np.testing.assert_array_equal(abc.data, np.array([[1,2,3]]))

    abc = ABC(*[np.ones(10) for _ in range(3)])
    np.testing.assert_array_equal(abc.data, np.ones((10,3)))

    with raises(TypeError):
        abc = ABC([1,2,3], [1,2], [1,2,3,4])

    abc = ABC([1,2], [1,2], [1,2])

    np.testing.assert_array_equal(abc.data, np.array([[1,2], [1,2], [1,2]]).T)

def test_init_kwargs():
    with raises(TypeError):
        abc = ABC(1,b=2,c=3)

    abc = ABC(a=1,b=2,c=3)
    np.testing.assert_array_equal(abc.data, np.array([[1,2,3]]))

    abc = ABC(a=[1,1],b=[1,1],c=[1,1])
    np.testing.assert_array_equal(abc.data, np.ones((2,3)))

    abc = ABC(data=np.ones((10,3)))
    np.testing.assert_array_equal(abc.data, np.ones((10,3)))

    with raises(TypeError):
        ABC(ggg=234342)

def test_init_list_of_self():
    abc = ABC([ABC(1,2,3), ABC(np.ones((4,3)))])
    np.testing.assert_array_equal(abc.b, np.array([2,1,1,1,1]))

def test_init_empty():
    with raises(TypeError):
        ABC()

def test_init_df():
    abc = ABC(pd.DataFrame(np.tile([1,2,3], (20,1)), columns=list("abc")))
    assert all(abc.c == 3)

def test_attr():
    abc = ABC(np.ones((5,3)))
    
    np.testing.assert_array_equal(abc.a, np.ones(5))
    np.testing.assert_array_equal(abc.b, np.ones(5))
    np.testing.assert_array_equal(abc.c, np.ones(5))
        
    assert dir(abc) == a_b_c

    with raises(AttributeError):
        d = abc.d


def test_getitem():
    abc = ABC(np.tile(np.linspace(0,4,5), (3,1)).T)

    assert abc[0] == ABC(np.array([[0,0,0]]))
    assert abc[0][0] == ABC(np.array([[0,0,0]]))

    assert abc[2:][0] == ABC(np.array([[2,2,2]]))



def test_eq():
    assert ABC(np.ones((5,3))) == ABC(np.ones((5,3)))

    assert not ABC(np.zeros((5,3))) == ABC(np.ones((5,3)))
    with raises(TypeError):
        ABC(np.zeros((5,3))) == ABC(np.zeros((6,3)))

    assert ABC(1,2,4) == ABC(1.0, 2.0, 4.0)
    
    assert ABC(np.ones((5,3))) == 1

def test_add():
    assert ABC(1,2,3) + 1 == ABC(2,3,4)
    assert 1 + ABC(1,2,3) == ABC(2,3,4)
    assert ABC(1,2,3) + ABC(1,2,3) == ABC(2,4,6)
    assert ABC(1,1,1) + np.ones(10) == ABC(np.full((10,3), 2))

def test_mul():
    assert ABC(1,2,3) * 2 == ABC(2,4,6)
    assert 2 * ABC(1,2,3) == ABC(2,4,6)
    assert ABC(1,2,3) * ABC(1,2,3) == ABC(1,4,9)
    assert ABC(2,2,2) * np.ones(10) == ABC(np.full((10,3), 2))

    a = ABC(1,1,1) * np.array([2])
    b = np.array([2]) * ABC(1,1,1)
    assert a.data.shape == b.data.shape

    assert ABC(1,1,1).tile(10) * np.ones(10) == ABC(np.ones((10,3)))

    assert ABC(1,2,3) * np.full(5, 2) == ABC(2,4,6).tile(5)

    assert ABC(1,2,3) * ABC(2,2,2).tile(10) == ABC(2,4, 6).tile(10)


def test_div():
    assert ABC(1,2,3) / 2 == ABC(0.5,1,1.5)
    assert 6 / ABC(1,2,3) == ABC(6,3,2)

    assert (6 / ABC(1,2,3)).data.shape == (1,3)


def test_abs():
    assert abs(ABC(1,1,1)) == np.sqrt(3)
    assert abs(ABC(2,2,2)) == np.sqrt(12)
    np.testing.assert_array_equal(abs(ABC(np.ones((5,3)))), np.full(5, np.sqrt(3)))


def test_diff():
    testarr = ABC(np.tile(np.linspace(0,100,20), (3,1)).T)
    np.testing.assert_array_almost_equal(
        testarr.diff(np.full(20,1)).data, 
        np.tile(np.full(20, 100/19), (3,1)).T
    ) 


def test_full():
    full =ABC(1,2,3).tile(100)

    assert len(full) == 100
    assert full[50] == ABC(1,2,3)



@mark.skip("not expected to work yet")
def test_single_col():
    class A(Base):
        cols = ["a"]

    assert A(1).data == np.array([[1]])


    np.testing.assert_array_equal(A([1,2,3]).data, np.array([[1],[2],[3]]))


def test_concatenate():
    a = ABC.full(ABC(1,2,3), 10)
    b = ABC.full(ABC(4,5,6), 10)
    c=ABC.concatenate([a,b])

    assert len(c) == len(a) + len(b) 
    np.testing.assert_array_equal(c.a, np.concatenate([a.a, b.a], axis=0))



def test_repr__():
    p = ABC(1,2,3)
    rpr = p.__repr__().split("\n")
    assert rpr[0] == "ABC"
    assert np.all(rpr[1:] == str(p.to_pandas()).split("\n"))
    