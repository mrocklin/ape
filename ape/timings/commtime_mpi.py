import ast
import os
import numpy as np

ape_dir = '/home/mrocklin/workspace/ape/'

def commtime_dict_mpi(network, nbytes=[10, 100, 1000, 10000]):
    if not all(v['type'] == 'mpi' for v in network.values()):
        raise ValueError("Trying to compute network properties of non-mpi"
                         "connections")
    # TODO: This is incorrect. We're assuming that the network is a clique
    hosts = set(host for (send, recv) in network for host in (send, recv))

    performance = model_dict_group(comm_times_group(nbytes, hosts))

    # inject new information into network dict
    return {key: merge(network[key], performance[key]) for key in network}

def comm_times_single(ns, send_host, recv_host):
    """ Computes transit times between two hosts
    Returns a list of [(nbytes, transit-time)]
    >>> comm_times([10, 100, 1000], 'sender.univ.edu', 'receiver.univ.edu')
     [(4, 0.00013303756713867188),
     (40, 0.00015091896057128906),
     (400, 0.0002040863037109375),
     (4000, 0.0005209445953369141)]
    """

    hosts = (send_host, recv_host)
    file = open('_machinefile.txt', 'w')
    file.write('\n'.join(hosts))
    file.close()
    s = os.popen('''mpiexec -np 2 -machinefile _machinefile.txt python %sape/timings/commtime_mpi_run_single.py "%s" %s %s'''%(ape_dir, ns, hosts[0], hosts[1]))

    values = ast.literal_eval(s.read())
    return values

def comm_times_group(ns, hosts):
    """ Computes transit times between a set of hosts
    Returns a list of [(sender, receiver, nbytes, transit-time)]
    >>> comm_times([10, 100, 1000], 'A', 'B', 'C')
    todo
    """

    file = open('_machinefile.txt', 'w')
    file.write('\n'.join(hosts))
    file.close()
    s = os.popen('''mpiexec -np %d -machinefile _machinefile.txt python %sape/timings/commtime_mpi_run_group.py "%s" %s'''%(len(hosts), ape_dir, ns, ' '.join(hosts))).read()

    values = ast.literal_eval(s)
    return values

def model_from_values(bytes_times):
    """ Given results of comm_times_single produce an intercept and slope

    inputs [(nbytes, transit_time)]

    outputs (intercept, slope)
    """
    nbytes, times = zip(*bytes_times)
    slope, intercept = np.polyfit(nbytes, times, 1)
    return {'intercept': intercept, 'slope':slope}


def model_dict_group(values):
    """ Converts data from comm_times_group into a dict of intercept/slopes

    inputs  - list of data :: [(sender, receiver, nbytes, duration)]
    outputs - dict mapping :: {sender, receiver : (intercept, slope)}
    """
    hosts = set(sender for sender, _, _, _ in values)
    nbytes_set = set(nbytes for _, _, nbytes, _ in values)
    data = dict(map(lambda (a,b,c,d) : ((a,b,c), d), values))
    data = {(sender, receiver, nbytes): duration for
            sender, receiver, nbytes, duration in values}

    return {(sender, receiver) :
                model_from_values([(nbytes, data[sender, receiver, nbytes])
                                  for nbytes in nbytes_set])
             for sender in hosts
             for receiver in hosts
             if sender != receiver}

def function_from_group_dict(d):
    """ Create function to compute communication times given int/slope data

    inputs -- Dictionary mapping :: {sender, receiver : intercept, slope}
    outputs -- Callable function :: nbytes, sender, receiver -> time (float)

    See also
        model_dict_group (produces input)
    """
    def commtime(nbytes, sender, receiver):
        """ Approximates communication time between sender and receiver """
        intercept, slope = d[sender, receiver]
        return nbytes*slope + intercept
    return commtime

def model(ns, send_host, recv_host):
    """ Computes the latency and inverse bandwidth between two hosts

    returns latency (intercept) and inverse bandwidth (slope) of the values
    returned by comm_times

    >>> model([10, 100, 1000], 'sender.univ.edu', 'receiver.univ.edu')
    (0.00017246120293471764, 8.8186875357188027e-08)

    time = .000172 + 8.818e-8*nbytes

    TODO - this function minimizes squared error. The larger values will
    dominate. Should weight nbytes = 10 similarly to nbytes = 1e9
    """

    values = comm_times_single(ns, send_host, recv_host)
    nbytes, times = zip(*values)
    slope, intercept = np.polyfit(nbytes, times, 1)
    return intercept, slope

def make_commtime_function(intercept, slope):
    def commtime(nbytes):
        return intercept + nbytes*slope
    return commtime

