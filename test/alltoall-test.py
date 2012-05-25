"""
We need to communicate hostnames to all processors
We use the mpi4py alltoall function

usage
mpiexec -np 3 python alltoall-test.py
"""

from mpi4py import MPI
import numpy as np

def host_name():
    import os
    return os.popen('uname -n').read().strip()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = MPI.COMM_WORLD.Get_size()
host = host_name()

names = comm.alltoall([host]*num_procs)
print names
