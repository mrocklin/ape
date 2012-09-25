from cpu import comptime_dict_cpu
from gpu import comptime_dict_gpu
from ape.util import merge

comptime_dict_fns = [comptime_dict_cpu, comptime_dict_gpu]

def comptime_dict(i, o, input_shapes, niter, machines, machine_groups=None):
    """ Estimate record average computation times of tasks in a graph

    inputs:
        i               - a theano.FunctionGraph describing the computation
        o               - a theano.FunctionGraph describing the computation
        input_shapes    - a dict {var: (shape)} for each input variable
        niter           - the number of times to run each computation
        machines        - a list of machines on which to run each computation
        machine_groups  - an iterable of sets of identical machines
                        - only a representative of each set will be used

    outputs:
        A dict mapping {{set-of-identical-machines}: {apply-node : runtime}}

    See Also:
        make_runtime_function   - converts the output of this function into a
                                - callable function
    """
    dicts = (fn(i, o, input_shapes, niter, machines, machine_groups)
             for fn in comptime_dict_fns)
    return merge(*dicts)

def make_runtime_function(machine_time_dict):
    """
    Create a callable function from a dict containing runtimes for machines

    inputs:
        machine_time_dict - dict mapping set of machines to times Dict
            each times dict maps string representation of an apply node to a
            time
    example-input:

    >>> {('ankaa', 'mimosa'):
            {'Elemwise{add,no_inplace}(x, Elemwise{mul,no_inplace}.0)': 0.01074,
             'Elemwise{mul,no_inplace}(x, x)': 0.005693912506103516},
         ('arroyitos', 'baconost'):
            {'Elemwise{add,no_inplace}(x, Elemwise{mul,no_inplace}.0)':0.01074,
             'Elemwise{mul,no_inplace}(x, x)': 0.005693912506103516}}

    output:
        A callable function that takes a machine-id (like 'ankaa') and an apply
        node and returns a time
    """
    # Break out the ids to a flat dict
    d = {id:machine_time_dict[ids] for ids in machine_time_dict for id in ids}

    def runtime_fn(apply, id):
        """ Given machine id and an apply node provides the expected runtime"""
        return d[id][str(apply)]

    return runtime_fn
