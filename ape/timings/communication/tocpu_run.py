# Computes transfer time from a GPU to a CPU

if __name__ == '__main__':
    import time
    from sys import argv, stdout
    from ape.theano_gpu_util import cpu_to_gpu_var, togpu_data
    import theano
    import numpy as np
    import ast

    ns = argv[1]
    ns = ast.literal_eval(ns)

    results = []

    # Make function to send variables from gpu to cpu
    x = theano.tensor.matrix('x')
    gx, cx = cpu_to_gpu_var(x)
    send = theano.function((gx,), cx)

    for n in ns:

        xx = np.ones((n,1), dtype=np.float32)
        gxx = togpu_data(xx)

        starttime = time.time()
        cxx = send(gxx)
        endtime = time.time()
        duration = endtime - starttime
        results.append((xx.nbytes, duration))

    stdout.write("[%s]\n"%(',\n'.join(map(str, results))))

