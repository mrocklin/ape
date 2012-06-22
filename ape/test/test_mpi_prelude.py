# If nose sees us
def test_mpi_send():
    import os
    # Run this file with two processors
    output = os.popen('mpiexec -np 2 -wdir /home/mrocklin/workspace/ape -machinefile /home/mrocklin/workspace/ape/ape/test/machinefile.txt python ape/test/test_mpi_prelude.py').read()
    assert output == "True"

import sys
import os
sys.path.insert(0,os.path.abspath("/home/mrocklin/workspace/ape/"))
# If we're being run by someone (hopefully mpiexec)
if __name__ == '__main__':

    from ape.mpi_prelude import *

    n = 1000
    x = np.ones(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)

    if host == 'mimosa.cs.uchicago.edu':
        send(x, 1, 'ankaa.cs.uchicago.edu')
        wait(1)
        recv(y, 2, 'ankaa.cs.uchicago.edu')
        wait(2)
        x = x+y
    if host == 'ankaa.cs.uchicago.edu':
        recv(y, 1, 'mimosa.cs.uchicago.edu')
        wait(1)
        x = x + y
        send(x, 2, 'mimosa.cs.uchicago.edu')

    if host=='ankaa.cs.uchicago.edu':
        sys.stdout.write(str((x==2).all()))

