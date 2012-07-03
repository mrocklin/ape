from ape.util import iterable, chain, load_dict, save_dict

def test_iterable():
    assert iterable([1,2])
    assert not iterable(3)

def test_chain():
    f = lambda x: x**2
    g = lambda x: (x, x+2)
    h = lambda x,y: (x+y)
    k = chain(g, h, f)
    assert k(2) == 36

def test_save_dict():
    data = {('A', 'B'): {'gemm':1, 'sum':2},
            ('C',)    : {'gemm':3, 'sum':1}}
    save_dict('_temp.tmp', data)
    data2 = load_dict('_temp.tmp')
    assert data == data2

