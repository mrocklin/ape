# See companion file mpi_computation_time.py

if __name__ == '__main__':
    import time
    from sys import argv, stdout, stdin, stderr
    import ast
    import numpy as np
    from ape.env_manip import unpack
    import theano
    from ape.env_manip import fgraph_iter
    from theano.tensor.utils import shape_of_variables

    stderr.write("Read in input shapes\n")
    input_shapes_str = argv[1]
    input_shapes = ast.literal_eval(input_shapes_str)

    niter = int(argv[2])

    stderr.write("Read in fgraphstr\n")
    fgraphstr = stdin.read()
    fgraph = unpack(fgraphstr)
    varof = {var.name: var for var in fgraph.variables}

    known_shapes = shape_of_variables(fgraph,
                                      {varof[name]: val
                                        for name, val in input_shapes.items()})
    known_shapes = {str(k):v for k,v in known_shapes.items()}

    stderr.write(str(known_shapes))
    result = {}
    for fg in fgraph_iter(fgraph):
        stderr.write("\n\n%s\n\n\n"%str(fg))

        inputs = filter(lambda x: not isinstance(x, theano.Constant),
                        fg.inputs)
        # Compile and instantiate
        num_inputs = [np.random.rand(*known_shapes[str(var)]).astype(var.dtype)
                             for var in inputs]

        f = theano.function(inputs, fg.outputs)
        starttime = time.time()

        stderr.write("Computing %s\n"%(str(fg)))
        for n in xrange(niter):
            outputs = f(*num_inputs)

        endtime = time.time()
        duration = endtime - starttime

        result[str(fg)] = duration / niter

    stdout.write(str(result))
    stdout.close()
