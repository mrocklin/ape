from theano import *
import theano.tensor as T
from theano_util import intermediate_shapes
from theano_computation import *

fast_run = theano.compile.optdb.query(theano.gof.Query(include = ['fast_run']))
fast_run_cpu_only = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude=['gpu']))
fast_run_no_inplace = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude = ['inplace', 'gpu']))
cpu_mode = theano.Mode(optimizer=fast_run_no_inplace, linker='py')

def make_computation(n=6):
    xs = [T.matrix('x_%d'%i) for i in range(n)]
    ys, zs = [],[]
    for i, x in enumerate(xs):
        y = T.dot(x,x); y.name = 'y_%d'%i
        z = y.sum(); z.name = 'z_%d'%i
        ys.append(y); zs.append(z)

    w = sum(zs); w.name = 'w'

    f = theano.function(xs, w, mode=cpu_mode)
    shapes = intermediate_shapes(xs, [w], [(3000,3000)]*n)
    computation = TheanoComputation(f, [(3000,3000)]*n)

    def tuplify_shape(shape):
        #if len(shape)==0:   return (1,)
        #else:               return tuple(shape)
        return tuple(shape)
    shapes = {key.name:tuplify_shape(value) for key, value in shapes.items()}

    TheanoArrayVariable.known_shapes = shapes

    return computation
