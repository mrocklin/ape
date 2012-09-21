from ape.mpi_prelude import *
from ape.codegen import read_fgraph, read_inputs
import theano

print "Hello from ", host
filename_root = 'tmp/'+host

# Unpack envs/jobs from file
fgraph = read_fgraph(filename_root+".fgraph")

# Set up the compiler
from theano.gof.sched import sort_schedule_fn
from theano.tensor.io import mpi_cmps
# TODO: job_schedule = read_schedule(filename_root+".schedule")
# TODO: scheduler = explicit_schedule_fn(job_schedule)
scheduler = sort_schedule_fn(*mpi_cmps) # TODO: remove
linker = theano.OpWiseCLinker(schedule=scheduler)
mode = theano.Mode(linker=linker, optimizer=None)

# Compile
inputs, outputs = theano.gof.graph.clone(fgraph.inputs, fgraph.outputs)
f = theano.function(inputs, outputs, mode=mode)
print "\nCompilation on "+host+" finished"
for node in f.maker.linker.make_all()[-1]:
    print node

# Initialize variables
inputs = read_inputs(filename_root+".inputs")
print "\nInitialization on "+host+" finished"

# Wait for everyone to finish compiling and setting up receives
comm.barrier()

print "\n"
# Compute
import time
comm.barrier()
starttime = time.time()
print "Computation on "+host+" begun"
outputs = f(*inputs)
print "Computation on "+host+" finished"
comm.barrier()
endtime = time.time()
if rank == 0:
    print "Duration: %f"%(endtime - starttime)
