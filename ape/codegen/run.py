from ape.mpi_prelude import *
import theano

# Unpack envs/jobs from file
from ape.env_manip import unpack
filename = host+".fgraph"
file = open(filename, 'r')
fgraph = unpack(file); file.close()

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
import_string = "from %s import inputs"%host.replace('.', '_')
exec(import_string)
print "Initialization on "+host+" finished"

# Wait for everyone to finish compiling and setting up receives
comm.barrier()

print "Computation on "+host+" begun"
# Compute
outputs = f(*inputs)
print "Computation on "+host+" finished"


