from ape.mpi_prelude import *
import theano
from numpy import array

# Unpack envs/jobs from file
from ape.env_manip import unpack_many
env_file = open("%(env_filename)s", 'r')
envs = unpack_many(env_file)
env_file.close()

# Compile Theano Functions
mode = theano.compile.mode.get_default_mode()
%(compile)s

print "Compilation on "+host+" finished"

# Initialize variables
%(variable_initialization)s

print "Initialization on "+host+" finished"

# Non-blocking receives
%(recv)s

# Wait for everyone to finish compiling and setting up receives
comm.barrier()

print "Computation on "+host+" begun"

# Compute
%(compute)s

