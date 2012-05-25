#!/home/mrocklin/Software/epd-7.2-1-rh5-x86/bin/python

"""
We need to communicate hostnames to all processors
We use the mpi4py alltoall function
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

ranks = comm.alltoall([host]*num_procs)
print ranks


