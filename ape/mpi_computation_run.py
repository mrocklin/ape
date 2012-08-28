# See companion file mpi_computation_time.py

if __name__ == '__main__':
    import time
    from sys import argv, stdout, stdin, stderr
    import ast
    import numpy as np
    from ape.env_manip import unpack
    import theano

    # stderr.write("Read in input shapes\n")
    input_shapes_str = argv[1]
    input_shapes = ast.literal_eval(input_shapes_str)

    niter = int(argv[2])

    # stderr.write("Read in fgraphstr\n")
    fgraphstr = stdin.read()
    fgraph = unpack(fgraphstr)

    # stderr.write("Form inputs\n")
    inputs = [np.random.rand(*input_shapes[str(var)]).astype(var.dtype)
                for var in fgraph.inputs]

    # Compile
    f = theano.function(fgraph.inputs, fgraph.outputs)

    starttime = time.time()

    for n in xrange(niter):
        outputs = f(*inputs)

    endtime = time.time()
    duration = endtime - starttime

    stdout.write("%f"%duration)
    stdout.close()
