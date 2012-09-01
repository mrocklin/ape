import theano
from ape.timings.timings import make_commtime_function

def test_make_commtime_function():
    data = {('a','b'): (1, 1) , ('b','a'): (0, 10)}

    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = theano.tensor.dot(x, x) + y[:,0].sum() - x*y

    env = theano.FunctionGraph([x,y], [z])

    dot = env.toposort()[2]

    from theano.tensor.utils import shape_of_variables
    known_shapes = shape_of_variables(env, {x:(100,100), y:(100,100)})

    commtime = make_commtime_function(data, known_shapes)

    assert commtime(dot, 'a', 'b') == 4*100*100 * 1  + 1
    assert commtime(dot, 'b', 'a') == 4*100*100 * 10 + 0
