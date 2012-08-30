# See companion file mpi_computation_time.py

def debugprint(s):
    pass
    stderr.write(s+"\n")

if __name__ == '__main__':
    import time
    from sys import argv, stdout, stdin, stderr
    import ast
    import numpy as np
    from ape.env_manip import unpack_many
    import theano
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
        # Compile and instantiate
        num_inputs = [np.random.rand(*known_shapes[str(var)]).astype(var.dtype)
                             for var in inputs]

        f = theano.function(inputs, fgraph.outputs)
        starttime = time.time()

        debugprint("Computing")
        for n in xrange(niter):
            outputs = f(*num_inputs)

        endtime = time.time()
        duration = endtime - starttime

        results.append(duration / niter)

    stdout.write(str(results))
    stdout.close()
