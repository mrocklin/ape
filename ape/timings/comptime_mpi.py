from ape import ape_dir
from ape.env_manip import pack_many, env_with_names
from ape.env_manip import fgraph_iter
from theano.tensor.utils import shape_of_variables
import os
import ast
from ape.util import dearrayify

def comptime_dict_mpi(fgraph, input_shapes, niter, machine_groups):
    return {mg: compute_time_on_machine(fgraph, input_shapes, mg[0], niter)
            for mg in machine_groups}

def compute_time_on_machine(fgraph, input_shapes, machine, niter):
    """ Computes computation time of funciton graph on a remote machine

    Returns average duration of the computation (time)

    inputs:
        fgraph  - A Theano FunctionGraph
        input_shapes - A dict mapping input variable to array shape
        machine - A machine on which to run the graph
        niter - The number of times to run the computation in sampling

    outputs:
        A dict mapping apply node to average runtime

    >>> compute_time_on_machine(fgraph, {x: (10, 10)}, 'receiver.univ.edu', 10)
    {dot(x, x+y): 0.133, Add(x, y): .0012}
    """

    file = open('_machinefile.txt', 'w')
    file.write(machine)
    file.close()

    # stringify the keys

    if len(set(map(str, fgraph.variables))) != len(fgraph.variables):
        raise ValueError("Not all variables have unique names"
                         "Look into ape.env_manip.variables_with_names")

    known_shapes = dearrayify(shape_of_variables(fgraph, input_shapes))

    known_shapes_str = str({str(k):v for k,v in known_shapes.items()})

    stdin, stdout, stderr = os.popen3('''mpiexec -np 1 -machinefile _machinefile.txt python %sape/timings/comptime_run.py "%s" %d'''%(ape_dir, known_shapes_str, niter))

    # Send the fgraphs as strings (they will be unpacked on the other end)

    fgraphs = fgraph_iter(fgraph)
    pack_many(fgraphs, stdin) # This writes to stdin
    stdin.close() # send termination signal

    # Receive the output from the compute node
    # return stdout.read() + stderr.read()
    message = stdout.read()
    times = ast.literal_eval(message)
    return  dict(zip(map(str, fgraph.nodes), times))
