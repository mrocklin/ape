import theano
import numpy as np

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

    profmode = theano.ProfileMode(optimizer=None,
            linker=theano.gof.OpWiseCLinker())
    f = theano.function(inputs, outputs, mode=profmode)

    numeric_inputs = [np.empty(input_shapes[var], dtype=var.dtype)
                                                   for var in inputs]
    for i in range(niter):  f(*numeric_inputs)

    stats = profmode.profile_stats[f].apply_time

    avg_time = {str(an):stats[an]/niter for an in stats}

    def runtime_of(an, id):
        """ Approximates the runtime of running Apply node an on machine id """
        if not valid_machine(id):
            raise NotImplementedError(
                    "This function is not valid for this machine")

        return avg_time[str(an)]

    return runtime_of
