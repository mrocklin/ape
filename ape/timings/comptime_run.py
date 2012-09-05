# See companion file mpi_computation_time.py
from sys import argv, stdout, stdin, stderr
import theano
import numpy as np
from ape.env_manip import unpack_many
import ast

def debugprint(s):
    pass
    # stderr.write(str(s)+"\n")

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
