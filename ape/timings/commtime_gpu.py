
def commtime_dict_togpu(network, nbytes=[10, 100, 1000, 10000]):
    if not all(v['type'] == 'togpu' for v in network.values()):
        raise ValueError("Trying to compute network properties of incompatible"
                         "network - must contain only togpu connections")
    raise NotImplementedError()

def commtime_dict_fromgpu(network, nbytes=[10, 100, 1000, 10000]):
    if not all(v['type'] == 'fromgpu' for v in network.values()):
        raise ValueError("Trying to compute network properties of incompatible"
                         "network - must contain only fromgpu connections")
    raise NotImplementedError()
