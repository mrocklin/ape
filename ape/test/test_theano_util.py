from ape.theano_util import bytes_of_dtype, shape_of_variables
import numpy as np
import theano

def test_bytes_of_dtype():
    assert bytes_of_dtype('float32') == 4

def test_shape_of_variables():
    x = theano.tensor.matrix('x')
    y = x[1:, 2:]
    known_shapes = shape_of_variables((x,), (y,), {x: (10, 10)})
    assert known_shapes[y] == (9, 8)
    assert not isinstance(known_shapes[y][0], np.ndarray)

    # just make sure that we can do this afterwards
    fgraph = theano.FunctionGraph((x,), (y,))
