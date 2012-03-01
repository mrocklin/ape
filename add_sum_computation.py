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

def make_computation(n, input_shape):
    xs = [T.matrix('x_%d'%i) for i in range(n)]
    ys, zs = [],[]
    for i, x in enumerate(xs):
        y = x+2; y.name = 'y_%d'%i
        z = y.sum(); z.name = 'z_%d'%i
        ys.append(y); zs.append(z)

    w = sum(zs); w.name = 'w'

    f = theano.function(xs, w, mode=cpu_mode)
    shapes = f.maker.env.shape_feature.shape_of

    shapes = intermediate_shapes(xs, [w], [input_shape]*n)
    computation = TheanoComputation(f, [input_shape]*n)

    return computation
