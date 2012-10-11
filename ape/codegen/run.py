from ape.mpi_prelude import *
from ape.codegen import read_graph, read_inputs, read_sched, sched_to_cmp
import theano
from sys import argv

rootdir = argv[1]
filename_root = rootdir+host

# Unpack envs/jobs from file
print filename_root+".sched"
graph = read_graph(filename_root+".fgraph")

# Set up the compiler
from theano.gof.sched import sort_schedule_fn
from theano.tensor.io import mpi_cmps

sched_cmp = sched_to_cmp(read_sched(filename_root+".sched"))

# Gather gpu comparators if we have a GPU
if theano.config.device == 'gpu':
    from theano.sandbox.cuda.async import gpu_cmps
    gpu_sched_cmp = sched_to_cmp(read_sched(filename_root+"-gpu.sched"))
    gpu_cmps = (gpu_sched_cmp,) + gpu_cmps
else:
    gpu_cmps = ()

scheduler = sort_schedule_fn(sched_cmp, *(gpu_cmps + mpi_cmps))
linker = theano.OpWiseCLinker(schedule=scheduler)
mode = theano.Mode(linker=linker, optimizer=None)

# Compile
inputs, outputs = theano.gof.graph.clone(graph.inputs, graph.outputs)
f = theano.function(inputs, outputs, mode=mode)

text = "\nCompilation on "+host+" finished"
for node in f.maker.linker.make_all()[-1]:
    text += str(node) + str('\n')
print text

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
