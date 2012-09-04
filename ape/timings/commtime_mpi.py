import numpy as np
from ape.util import merge
from ape import ape_dir
from ape.timings.commtime_util import run_on_hosts, model_dict_group

def commtime_dict_mpi(network, nbytes=[10, 100, 1000, 10000]):
    """
    inputs
        network - dict like {(A, B): {'type': 'mpi'}}
        nbytes  - iterable of byte counts

    outputs
        network - dict like {(A, B): {'type': 'mpi', 'intercept':1, 'slope':2}}
    """
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

    return run_on_hosts((send_host, recv_host),
        '''python %sape/timings/commtime_mpi_run_single.py "%s" %s %s'''%(
            ape_dir, str(ns), send_host, recv_host))

def comm_times_group(ns, hosts):
    """ Computes transit times between a set of hosts
    Returns a list of [(sender, receiver, nbytes, transit-time)]
    >>> comm_times([10, 100, 1000], 'A', 'B', 'C')
    todo
    """

    return run_on_hosts(hosts,
        '''python %sape/timings/commtime_mpi_run_group.py "%s" %s'''%(
            ape_dir, ns, ' '.join(hosts)))
