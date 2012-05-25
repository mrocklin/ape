import os
def test_mpi_send():
    assert os.popen('mpiexec -np 2 python mpi_env_send.py').read() == "True"

