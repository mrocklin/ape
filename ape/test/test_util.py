from ape.util import (iterable, chain, load_dict, save_dict, dearrayify, merge,
        unique)

def test_iterable():
    assert iterable([1,2])
    assert not iterable(3)

def test_chain():
    f = lambda x: x**2
    g = lambda x: (x, x+2)
    h = lambda x,y: (x+y)
    k = chain(g, h, f)
    assert k(2) == 36

def test_chain_iterable_inputs():
    f = lambda x : (x, x+2)
    g = lambda L : sum(L)
    h = chain(f, g)
    k = chain(f, sum)
    assert h(2) == 6
    assert k(2) == 6

def test_save_dict():
    data = {('A', 'B'): {'gemm':1, 'sum':2},
            ('C',)    : {'gemm':3, 'sum':1}}
    save_dict('_temp.tmp', data)
    data2 = load_dict('_temp.tmp')
    assert data == data2

def test_dearrayify():
    from numpy import array
    assert dearrayify(3) == 3
    assert dearrayify(array(3)) == 3
    assert dearrayify((3, array(4))) == (3, 4)
    assert dearrayify((3, 4)) == (3, 4)
    assert dearrayify({'x': (array(1000), array(1000)), 'y': (), 'z': 5}) == \
            {'x': (1000, 1000), 'y': (), 'z': 5}

def test_merge():
    d = {1:2, 3:4}
    e = {4:5}
    assert merge(d, e) == {1:2, 3:4, 4:5}

def test_unique():
    assert unique('abc')
    assert not unique('aab')

def test_remove():
    assert remove(lambda x: x > 5, range(10)) == [1, 2, 3, 4, 5]
