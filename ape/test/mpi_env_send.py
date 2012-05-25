import sys
sys.path.append('..')

from env_manip import *
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = MPI.COMM_WORLD.Get_size()

if rank == 0:
    x = T.matrix('x')
    y = x+x*x
    env = theano.Env([x], [y])
    s = pack(env)
    comm.send(s, dest=1, tag = 1234)

if rank == 1:
    s = comm.recv(source = 0, tag = 1234)
    env = unpack(s)
    f = theano.function(env.inputs, env.outputs[0])
    sys.stdout.write(str(f(np.ones((5,5))).sum() == 50))
