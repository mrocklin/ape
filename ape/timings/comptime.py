from comptime_mpi import comptime_dict_mpi
from comptime_gpu import comptime_dict_gpu

def make_runtime_function(machine_time_dict):
    """
    Create a callable function from a dict containing runtimes for machines

    inputs
    ------
    machine_time_dict : Dict mapping set of machines to times Dict
        each times dict maps string representation of an apply node to a time
        example:
    >>> {('ankaa', 'mimosa'):
            {'Elemwise{add,no_inplace}(x, Elemwise{mul,no_inplace}.0)': 0.01074,
             'Elemwise{mul,no_inplace}(x, x)': 0.005693912506103516},
         ('arroyitos', 'baconost'):
            {'Elemwise{add,no_inplace}(x, Elemwise{mul,no_inplace}.0)':0.01074,
             'Elemwise{mul,no_inplace}(x, x)': 0.005693912506103516}}

    outputs
    -------
    A callable function that takes a machine-id (like 'ankaa') and an apply
    node and returns a time
    """
    # Break out the ids to a flat dict
    d = {id:machine_time_dict[ids] for ids in machine_time_dict for id in ids}

    def runtime_fn(apply, id):
        """ Given machine id and an apply node provides the expected runtime"""
        return d[id][str(apply)]

    return runtime_fn

