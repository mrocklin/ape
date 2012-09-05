import numpy as np

def model_from_values(bytes_times):
    """ Given results of comm_times_single produce an intercept and slope

    inputs [(nbytes, transit_time)]

    outputs {'intercept': .001, 'slope': .000054}
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

def make_commtime_function(intercept, slope):
    def commtime(nbytes):
        return intercept + nbytes*slope
    return commtime

