import theano
from ape.timings.communication.master import make_commtime_function, commtime_dict
from theano.tensor.utils import shape_of_variables

x = theano.tensor.matrix('x')
y = theano.tensor.matrix('y')
z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y

env = theano.FunctionGraph([x,y], [z])

dot = env.toposort()[2]
known_shapes = shape_of_variables(env, {x:(100,100), y:(100,100)})

def test_make_commtime_function():
    data = {('a','b'): {'intercept':1, 'slope':1},
            ('b','a'): {'intercept':0, 'slope':10}}

    commtime = make_commtime_function(data, known_shapes)

    assert commtime(dot, 'a', 'b') == 4*100*100 * 1  + 1
    assert commtime(dot, 'b', 'a') == 4*100*100 * 10 + 0

def test_make_commtime_function_non_clique_network():
    data = {('a','b'): {'intercept':1, 'slope':1},
            ('b','a'): {'intercept':0, 'slope':10},
            ('b','c'): {'intercept':0, 'slope':10}}

    commtime = make_commtime_function(data, known_shapes)

    assert commtime(dot, 'a', 'b') == 4*100*100 * 1  + 1
    assert commtime(dot, 'b', 'a') == 4*100*100 * 10 + 0
    assert commtime(dot, 'c', 'a') > 10000

def test_commtime_dict():
    network = {
        ('ankaa.cs.uchicago.edu', 'baconost.cs.uchicago.edu'): {'type': 'mpi'},
        ('baconost.cs.uchicago.edu', 'ankaa.cs.uchicago.edu'): {'type': 'mpi'},
        ('baconost.cs.uchicago.edu', 'baconostgpu'): {'type': 'togpu'},
        ('baconostgpu', 'baconost.cs.uchicago.edu'): {'type': 'fromgpu'}}

    result = commtime_dict(network, nbytes=[10, 100, 1000]*3)
    assert set(result.keys()) == set(network.keys())
    assert all('intercept' in val for val in result.values())
