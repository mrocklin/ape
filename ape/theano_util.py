import theano
from ape.util import dearrayify

known_dtypes = {'float32': 4, 'float64':8, 'int32':4, 'int64':8}
def bytes_of_dtype(dtype):
    if dtype in known_dtypes:
        return known_dtypes[dtype]
    raise NotImplementedError("Don't know how many bytes are in dtype%s"%dtype)

# This should be moved upstream
def shape_of_variables(i, o, input_shapes):
    fgraph = theano.FunctionGraph(i, o)
    known_shapes = dearrayify(
            theano.tensor.utils.shape_of_variables(fgraph, input_shapes))
    fgraph.disown()
    return known_shapes
