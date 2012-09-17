# Computes transfer time from a CPU to a GPU

if __name__ == '__main__':
    import time
    from sys import argv, stdout
    from ape.timings.theano_gpu_util import cpu_to_gpu_var
    import theano
    import numpy as np
    import ast

    ns = argv[1]
    ns = ast.literal_eval(ns)

    results = []

    # Make function to send variables from cpu to gpu
    cx = theano.tensor.matrix('x')
    gx = theano.sandbox.cuda.gpu_from_host(cx)
    send = theano.function((cx,), gx)

    for n in ns:

        xx = np.ones((n,1), dtype=np.float32)

        starttime = time.time()
        gxx = send(xx)
        endtime = time.time()
        duration = endtime - starttime
        results.append((xx.nbytes, duration))

    stdout.write("[%s]\n"%(',\n'.join(map(str, results))))

