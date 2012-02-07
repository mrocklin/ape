from theano import tensor as T
import theano
tdp = theano.printing.debugprint
fast_run = theano.compile.optdb.query(theano.gof.Query(include = ['fast_run']))
fast_run_cpu_only = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude=['gpu']))
fast_run_no_inplace = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude = ['inplace', 'gpu']))
cpu_mode = theano.Mode(optimizer=fast_run_no_inplace, linker='py')

from infrastructure import CommNetwork, ComputationalNetwork, CommNetwork
from theano_infrastructure import (CPUWorker, GPUWorker, MPIWire, importall,
        CPUWireGPU, GPUWireCPU)
from theano_computation import (TheanoArrayVariable, TheanoJob, TheanoVariable,
        TheanoComputation, all_jobs)
from theano_to_milp import intermediate_shapes, make_ilp, compute_schedule

from IPython.parallel import Client

# Connect to machines
rc = Client(profile = 'mpi')
view = rc[:]
importall(view)
a,b,c = rc[0], rc[1], rc[2]
A,B,C = map(CPUWorker, (a,b,c))
machines = [A,B,C]
try: # add a gpu if we can
    G = GPUWorker(C)
    machines.append(G)
except:  pass

# Set up communication network
wires = [MPIWire(a,b) for a in [A,B,C] for b in [A,B,C] if a!=b]
try:    wires += [CPUWireGPU(C,G), GPUWireCPU(G,C)]
except: pass
network = CommNetwork(wires)

system = ComputationalNetwork(machines, network)

# Make a computation
xx = T.matrix('xx')
yy = T.dot(xx,xx); yy.name = 'yy'
zz = yy.sum(); zz.name = 'zz'
x = T.matrix('x')
y = T.dot(x,x); y.name = 'y'
z = y.sum(); z.name = 'z'
w = z+zz; w.name = 'w'
f = theano.function([x, xx], w, mode=cpu_mode)
# f = theano.function([x], z, mode=cpu_mode)

shapes = intermediate_shapes([x,xx], [w], [(3000,3000), (3000,3000)])
#shapes = intermediate_shapes([x,xx], [w], [(1000,1000), (500,500)])
#shapes = intermediate_shapes([x], [z], [(1000,1000)])

def tuplify_shape(shape):
    shape = tuple(shape)
    if shape==tuple():
        shape = (1,)
    return shape
shapes = {key.name:tuplify_shape(value) for key, value in shapes.items()}

#computation = TheanoComputation(f, [(1000,1000), (500,500)])
#computation = TheanoComputation(f, [(1000,1000)])
computation = TheanoComputation(f, [(1000,1000), (1000,1000)])

TheanoArrayVariable.known_shapes = shapes

prob, X, S, Cmax = make_ilp(computation, system, A, niter=3)
sched = compute_schedule(prob, X, S, Cmax)


def test_single_wire():
    an = f.maker.env.outputs[0].owner
    job = TheanoJob(an)
    V = TheanoArrayVariable(job.inputs[0]._variable, (1000,1000))
    w = MPIWire(A, B)
    A.instantiate_random_variable(V)
    A.compile(job)
    A.run(job)
    for output in job.outputs:
        w.transmit(output)

    assert all(output in B for output in job.outputs)

