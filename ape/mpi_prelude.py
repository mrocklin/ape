from mpi4py import MPI
import numpy as np

def host_name():
    import os
    return os.popen('uname -n').read().strip()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = host_name()
def exchange_ranks():
    names = comm.alltoall([host]*size)
    return dict(zip(names, range(comm.Get_size())))

# Perform handshake
rank_of_machine = exchange_ranks()

# Wrap MPI calls
requests = dict()
def send(var, tag, dest_machine_id):
    request = comm.Isend(var, rank_of_machine[dest_machine_id], tag)
    requests[tag] = request
def recv(var, tag, source_machine_id):
    request = comm.Irecv(var, rank_of_machine[source_machine_id], tag)
    requests[tag] = request
def wait(tag):
    requests[tag].wait()
