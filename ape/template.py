from mpi4py import MPI
import theano
from env_manip import unpack_many

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

# dictionary mapping variable_name to unique key
%(variable_tags)s

# Wrap MPI calls
send_requests = dict()
recv_requests = dict()
def send(var, dest_machine_id):
    request = comm.isend(var, rank_of_machine[dest_machine_id], tag_of[var])
    send_requests[var, dest_machine_id] = request
def recv(var, source_machine_id):
    request = comm.irecv(var, rank_of_machine[source_machine_id], tag_of[var])
    recv_requests[var, source_machine_id] = request
def wait_on_send(var, id):
    send_requests[var, id].wait()
def wait_on_recv(var, id):
    recv_requests[var, id].wait()

env_file = open("%(env_filename)s", 'r')
envs = unpack_many(env_file)
env_file.close()

mode = theano.compile.mode.get_default_mode()
%(compile)s


%(host_code)s
