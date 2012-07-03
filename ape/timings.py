import theano
import numpy as np
from ape.util import chain, prod
from ape.theano_util import bytes_of_dtype
import ape.mpi_timing as mpi_timing

def compute_runtimes(inputs, outputs, input_shapes, niter=10):
    """
    Compute runtimes of all apply nodes within an env by sampling

    inputs
    ------
    inputs  - input variables to a computation  :: [theano.Variable]
    outputs - output variables to a computation :: [theano.Variable]
    input_shapes - dictionary from input variables to shapes :: {Tensor : shape}
    niter - number of iterations to run computation :: int

    outputs
    -------

    dictionary mapping string-name of apply node to expected time

    >>> import theano
    >>> x = theano.tensor.matrix('x')
    >>> y = x+x*x
    >>> times = compute_runtimes([x], [y], {x:(1000,1000)})
    >>> env = theano.Env([x], [y])
    >>> an = env.toposort()[0]
    >>> times[str(an)]
    .0024525

    note - runtime compares the string representation of the apply node. This
    may fail.
    """
    profmode = theano.ProfileMode(optimizer=None,
            linker=theano.gof.OpWiseCLinker())
    f = theano.function(inputs, outputs, mode=profmode)

    numeric_inputs = [np.empty(input_shapes[var], dtype=var.dtype)
                                                   for var in inputs]
    for i in range(niter):  f(*numeric_inputs)

    stats = profmode.profile_stats[f].apply_time

    avg_time = {str(an):stats[an]/niter for an in stats}
    return avg_time


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

def make_runtime_fn(inputs, outputs, input_shapes, valid_machine, niter=10):
    """
    Creates a function to approximate the runtime of an applynode on a machine

    Does this by running the apply nodes on this machine many times.

    inputs
    ------
    inputs  - input variables to a computation :: [theano.Variable]
    outputs - output variables to a computation :: [theano.Variable]
    input_shapes - dictionary from input variables to shapes :: {Tensor : shape}
    valid_machine - for which machines is this function valid? :: id -> bool
    niter - number of iterations to run computation :: int

    outputs
    -------
    runtime_of - a function to approximate the runtime of an apply node

    >>> import theano
    >>> x = theano.tensor.matrix('x')
    >>> y = x+x*x
    >>> valid_machine = lambda id : id in {'ankaa', 'mimosa'}
    >>> runtime = make_runtime_fn([x], [y], {x:(1000,1000)}, valid_machine)
    >>> env = theano.Env([x], [y])
    >>> an = env.toposort()[0]
    >>> runtime(an, 'ankaa')
    .0024525
    >>> runtime(an, 'bellatrix')
    NotImplementedError( ... )

    note - runtime compares the string representation of the apply node. This
    may fail.
    """

    avg_time = compute_runtimes(inputs, outputs, input_shapes, niter)

    def runtime_of(an, id):
        """ Approximates the runtime of running Apply node an on machine id """
        if not valid_machine(id):
            raise NotImplementedError(
                    "This function is not valid for this machine")

        return avg_time[str(an)]

    return runtime_of

compute_commtimes = chain( mpi_timing.comm_times_group,
                           mpi_timing.model_dict_group)

def make_commtime_function(cdict, known_shapes):
    """ Create callable function from a dict of intercept/slopes, known shapes

    inputs
    ------
    cdict - dictionary mapping {sender, receiver : intercept, slope}
    input_shapes - dictionary from input variables to shapes :: {Tensor : shape}

    outputs
    -------
    commtime - function :: ApplyNode, Sender, Receiver -> time (float)
    """

    bytes_fn = mpi_timing.function_from_group_dict(cdict)

    def bytes(var):
        """ Compute the bytes that a theano variable holds """
        shape = known_shapes[var]
        return prod(shape)*bytes_of_dtype(var.dtype)

    def commtime(an, sender, receiver):
        """ Returns the communication time to transmit the outputs of an """
        nbytes = sum(map(bytes, an.outputs))
        intercept, slope = cdict[sender, receiver]
        return slope*nbytes + intercept

    return commtime
