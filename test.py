from theano import tensor as T
import theano
tdp = theano.printing.debugprint
fast_run = theano.compile.optdb.query(theano.gof.Query(include = ['fast_run']))
fast_run_cpu_only = theano.compile.optdb.query(
        theano.gof.Query(include = ['fast_run'], exclude=['gpu']))
cpu_mode = theano.Mode(optimizer=fast_run_cpu_only, linker='py')

from infrastructure_graph import (CPUWorker, GPUWorker, MPIWire, importall,
        CommNetwork, CPUWireGPU, GPUWireCPU)
from computation_graph import (TheanoArrayVariable, TheanoJob, TheanoVariable,
        all_jobs)
from theano_to_milp import intermediate_shapes

from IPython.parallel import Client

rc = Client(profile = 'mpi')
view = rc[:]
importall(view)
a,b,c = rc[0], rc[1], rc[2]
A,B,C = map(CPUWorker, (a,b,c))
try:
    G = GPUWorker(C)
except:
    pass

x = T.matrix('x')
y = T.dot(x,x); y.name = 'y'
z = y.sum(); z.name = 'z'


f = theano.function([x], z, mode=cpu_mode)
shapes = intermediate_shapes([x], [z], [(1000,1000)])
def tuplify_shape(shape):
    shape = tuple(shape)
    if shape==tuple():
        shape = (1,)
    return shape
shapes = {key.name:tuplify_shape(value) for key, value in shapes.items()}
TheanoArrayVariable.known_shapes = shapes
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

# Set up communication network
wires = [MPIWire(a,b) for a in [A,B,C] for b in [A,B,C] if a!=b]
try:    wires += [CPUWireGPU(C,G), GPUWireCPU(G,C)]
except: pass
N = CommNetwork(wires)
