from mpi_prelude import *
import theano

# Unpack envs/jobs from file
from env_manip import unpack_many
env_file = open("%(env_filename)s", 'r')
envs = unpack_many(env_file)
env_file.close()

# Compile Theano Functions
mode = theano.compile.mode.get_default_mode()
%(compile)s

# Initialize variables
%(variable_initialization)s

# Non-blocking receives
%(recv)s

# Wait for everyone to finish compiling and setting up receives
comm.barrier()

# Compute
%(compute)s

