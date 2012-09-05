from sys import argv, stdout, stdin, stderr
import theano
import time
from ape.timings.comptime_run import debugprint, collect_inputs, comptime_run

mode = theano.compile.mode.get_default_mode()
mode = mode.excluding('gpu')

def time_computation(inputs, outputs, numeric_inputs, niter):

    f = theano.function(inputs, outputs, mode=mode)

    starttime = time.time()
    debugprint("Computing")
    for n in xrange(niter):
        outputs = f(*numeric_inputs)
    endtime = time.time()
    duration = endtime - starttime

    return duration/niter

if __name__ == '__main__':

    known_shapes, niter, fgraphs = collect_inputs(argv, stdin)
    results = comptime_run(known_shapes, niter, fgraphs, time_computation)

    stdout.write(str(results))
    stdout.close()

