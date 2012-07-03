from ape.util import iterable, chain

def test_iterable():
    assert iterable([1,2])
    assert not iterable(3)

def test_chain():
    f = lambda x: x**2
    g = lambda x: (x, x+2)
    h = lambda x,y: (x+y)
    k = chain(g, h, f)
    assert k(2) == 36
