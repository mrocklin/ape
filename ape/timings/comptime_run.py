# See companion file mpi_computation_time.py
import time
from sys import argv, stdout, stdin, stderr
import theano


def debugprint(s):
    pass
    stderr.write(str(s)+"\n")

def time_computation(inputs, outputs, numeric_inputs, niter):

    f = theano.function(inputs, outputs)

    starttime = time.time()
    debugprint("Computing")
    for n in xrange(niter):
        outputs = f(*numeric_inputs)
    endtime = time.time()
    duration = endtime - starttime

    return duration/niter

if __name__ == '__main__':
    import ast
    import numpy as np
    from ape.env_manip import unpack_many
    from theano.tensor.utils import shape_of_variables

    # Setup
    debugprint("Read in input shapes")
    known_shapes_str = argv[1]
    debugprint(known_shapes_str)
    known_shapes = ast.literal_eval(known_shapes_str)
    debugprint("\n%s\n"%str(known_shapes))

    niter = int(argv[2])

    results = []
    for fgraph in unpack_many(stdin):
        debugprint("\n%s\n"%str(fgraph))

        inputs = filter(lambda x: not isinstance(x, theano.Constant),
                        fgraph.inputs)
        outputs = fgraph.outputs
        # Compile and instantiate
        num_inputs = [np.asarray(np.random.rand(*known_shapes[str(var)])).astype(var.dtype)
                             for var in inputs]

        duration = time_computation(inputs, outputs, num_inputs, niter)

        results.append(duration)

    stdout.write(str(results))
    stdout.close()
