# See companion file mpi_computation_time.py
import time
from sys import argv, stdout, stdin, stderr
import theano
import numpy as np
from ape.env_manip import unpack_many
import ast


def debugprint(s):
    pass
    # stderr.write(str(s)+"\n")

def time_computation(inputs, outputs, numeric_inputs, niter):

    f = theano.function(inputs, outputs)

    starttime = time.time()
    debugprint("Computing")
    for n in xrange(niter):
        outputs = f(*numeric_inputs)
    endtime = time.time()
    duration = endtime - starttime

    return duration/niter

def collect_inputs(argv, stdin):

    debugprint("Read in input shapes")
    known_shapes_str = argv[1]
    debugprint(known_shapes_str)
    known_shapes = ast.literal_eval(known_shapes_str)

    niter = int(argv[2])

    fgraphs = unpack_many(stdin)

    return known_shapes, niter, fgraphs

def comptime_run(known_shapes, niter, fgraphs, time_computation_fn):
    # Setup
    debugprint("\n%s\n"%str(known_shapes))

    results = []
    for fgraph in fgraphs:
        debugprint("\n%s\n"%str(fgraph))

        inputs = filter(lambda x: not isinstance(x, theano.Constant),
                        fgraph.inputs)
        outputs = fgraph.outputs
        # Compile and instantiate
        num_inputs = [np.asarray(np.random.rand(*known_shapes[str(var)])).astype(var.dtype)
                             for var in inputs]

        duration = time_computation_fn(inputs, outputs, num_inputs, niter)

        results.append(duration)

    return results

if __name__ == '__main__':

    known_shapes, niter, fgraphs = collect_inputs(argv, stdin)
    results = comptime_run(known_shapes, niter, fgraphs, time_computation)

    stdout.write(str(results))
    stdout.close()

