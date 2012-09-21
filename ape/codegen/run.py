from ape.mpi_prelude import *
from ape.codegen import read_fgraph, read_inputs
import theano

filename_root = 'tmp/'+host

# Unpack envs/jobs from file
fgraph = read_fgraph(filename_root+".fgraph")

# Set up the compiler
from theano.gof.sched import scheduler
from theano.tensor.io import mpi_cmps
# TODO: Read in explicit schedule
scheduler = sort_schedule_fn(*mpi_cmps)
linker = theano.OpWiseCLinker(schedule=scheduler)
mode = theano.Mode(linker=linker, optimizer=None)

# Compile
inputs, outputs = theano.gof.graph.clone(fgraph.inputs, fgraph.outputs)
f = theano.function(inputs, outputs, mode=mode)
print "Compilation on "+host+" finished"

# Initialize variables
inputs = read_inputs(filename_root+".inputs")
print "Initialization on "+host+" finished"

# Wait for everyone to finish compiling and setting up receives
comm.barrier()

print "Computation on "+host+" begun"
# Compute
outputs = f(*inputs)
print "Computation on "+host+" finished"


