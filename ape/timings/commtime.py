from ape.timings.commtime_mpi import commtime_dict_mpi
from ape.timings.commtime_gpu import commtime_dict_togpu, commtime_dict_fromgpu
from ape.timings.commtime_util import function_from_group_dict
from ape.util import prod
from ape.theano_util import bytes_of_dtype

commtime_dict_fns = (commtime_dict_mpi, commtime_dict_togpu,
                     commtime_dict_fromgpu)

def commtime_dict(network, *args, **kwargs):
    """
    inputs
        network - dict like {(A, B): {'type': 'mpi'}}

    outputs
        network - dict like {(A, B): {'type': 'mpi', 'intercept':1, 'slope':2}}
    """
    for fn in commtime_dict_fns:
        network = fn(network, *args, **kwargs)
    return network

def _commtime_dict_interface(network):
    """
    inputs
        network - dict like {(A, B): {'type': 'mpi'}}

    outputs
        network - dict like {(A, B): {'type': 'mpi', 'intercept':1, 'slope':2}}
    """
    pass


def make_commtime_function(cdict, known_shapes):
    """ Create callable function from a dict of intercept/slopes, known shapes

    inputs
    ------
    cdict - dictionary mapping {(sender, receiver) : {'intercept':i, 'slope':s}}
    input_shapes - dictionary from input variables to shapes :: {Tensor : shape}

    outputs
    -------
    commtime - function :: ApplyNode, Sender, Receiver -> time (float)
    """

    bytes_fn = function_from_group_dict(cdict)
    known_shapes = {str(key): known_shapes[key] for key in known_shapes}

    def bytes(var):
        """ Compute the bytes that a theano variable holds """
        shape = known_shapes[str(var)]
        return prod(shape)*bytes_of_dtype(var.dtype)

    def commtime(an, sender, receiver):
        """ Returns the communication time to transmit the outputs of an """
        if sender == receiver:
            return 0
        nbytes = sum(map(bytes, an.outputs))
        intercept = cdict[sender, receiver]['intercept']
        slope     = cdict[sender, receiver]['slope']
        return slope*nbytes + intercept

    return commtime

