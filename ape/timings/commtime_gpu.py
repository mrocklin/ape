from ape.timings.commtime_util import model_from_values
from ape.timings.comm_util import run_on_hosts
from ape.util import merge
from ape import ape_dir

def commtime_dict_togpu(network, nbytes=[10, 100, 1000, 10000]):
    return {(host, gpu):
                merge(network[host, gpu],
                      model_from_values(comm_times_togpu(nbytes, host)))
            for host, gpu in network
            if network[host, gpu]['type'] == 'togpu'}

def commtime_dict_fromgpu(network, nbytes=[10, 100, 1000, 10000]):
    return {(gpu, host):
            merge(network[gpu, host],
                  model_from_values(comm_times_fromgpu(nbytes, host)))
            for gpu, host in network
            if network[gpu, host]['type'] == 'fromgpu'}

def comm_times_togpu(nbytes, host):
    """ Computes transit times between host and gpu
    Returns a list of [(nbytes, transit-time)]
    >>> comm_times([10, 100, 1000], 'host.univ.edu')
     [(40, 0.00015091896057128906),
     (400, 0.0002040863037109375),
     (4000, 0.0005209445953369141)]
    """

    return run_on_hosts((host, ),
       '''python %sape/timings/commtime_togpu_run.py "%s"'''%(
           ape_dir, str(nbytes)))
def comm_times_fromgpu(nbytes, host):
    """ Computes transit times between gpu and host
    Returns a list of [(nbytes, transit-time)]
    >>> comm_times([10, 100, 1000], 'host.univ.edu')
     [(40, 0.00015091896057128906),
     (400, 0.0002040863037109375),
     (4000, 0.0005209445953369141)]
    """

    return run_on_hosts((host, ),
     '''python %sape/timings/commtime_tocpu_run.py "%s"'''%(
         ape_dir, str(nbytes)))


