"""
We need to communicate hostnames to all processors
We use the mpi4py alltoall function

usage
mpiexec -np 3 python alltoall-test.py
"""

from mpi4py import MPI

def host_name():
    import os
    return os.popen('uname -n').read().strip()

def exchange_ranks():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = MPI.COMM_WORLD.Get_size()
    host = host_name()
    names = comm.alltoall([host]*num_procs)
    return dict(zip(names, range(comm.Get_size())))

if __name__ == "__main__":
    print exchange_ranks()
