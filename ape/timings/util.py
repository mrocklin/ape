import ast
import os
import theano

def run_on_hosts(hosts, command):
    """ Computes transit times between two hosts
    Returns a list of [(nbytes, transit-time)]
    >>> comm_times([10, 100, 1000], 'sender.univ.edu', 'receiver.univ.edu')
     [(4, 0.00013303756713867188),
     (40, 0.00015091896057128906),
     (400, 0.0002040863037109375),
     (4000, 0.0005209445953369141)]
    """

    file = open('_machinefile.txt', 'w')
    file.write('\n'.join(hosts))
    file.close()
    s = os.popen('''mpiexec -np %d -machinefile _machinefile.txt %s'''%(
                    len(hosts), command))

    values = ast.literal_eval(s.read())
    return values

def fgraph_iter(fgraph):
    """ Returns iterator of atomic funciton graphs - really just apply nodes"""
    for node in fgraph.apply_nodes:
        nn = node.clone_with_new_inputs([inp.clone() for inp in node.inputs])
        yield theano.FunctionGraph(nn.inputs, nn.outputs)
