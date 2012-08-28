from env_manip import pack
from ape import ape_dir
import os

def compute_time_on_machine(fgraph, input_shapes, machine, niter):
    """ Computes computation time of funciton graph on a remote machine

    Returns average duration of the computation (time)

    inputs:
        fgraph  - A Theano FunctionGraph
        input_shapes - A dict mapping input variable to array shape
        machine - A machine on which to run the graph
        niter - The number of times to run the computation in sampling

    >>> compute_time_on_machine(fgraph, {x: (10, 10)}, 'receiver.univ.edu', 10)
    .0133
    """

    file = open('_machinefile.txt', 'w')
    file.write(machine)
    file.close()

    fgraphstr = pack(fgraph)

    # stringify the keys
    input_shapes_str = str({str(k):v for k,v in input_shapes.items()})

    stdin, stdout, stderr = os.popen3('''mpiexec -np 1 -machinefile _machinefile.txt python %sape/mpi_computation_run.py "%s" %d'''%(ape_dir, input_shapes_str, niter))

    # Send the fgraph as a string (it will be unpacked on the other end)
    stdin.write(fgraphstr)
    stdin.close()

    # Receive the output from the compute node
    message = stdout.read()
    duration = float(message)
    return duration / niter
