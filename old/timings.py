from theano_infrastructure import GPUWorker

maclab = {"ankaa", "mimosa", "bellatrix"}
maclab = {name+".cs.uchicago.edu" for name in maclab}
def machine_type_id(w):
    if hasattr(w, "_machine_id"): # cache this value on the worker
        return w._machine_id
    name = w.get_hostname()
    if name in maclab:
        name = "maclab"
    if isinstance(w, GPUWorker):
        name += "_gpu"
    w._machine_id = name
    return name

def make_runtime_fn(computation, system, niter=3):
    """
    Compute runtimes of a computation on a system

    Returns a dict mapping (job, machine) -> runtime of job on machine

    This dict is actually a function disguised as a dict with caching enabled
    """
    jobs = computation.jobs
    machines = system.machines
    N = system.comm
    cache = dict()
    def runtime(job, machine):
        key = (job, machine_type_id(machine))
        if key in cache:
            value = cache[key]
        else:
            value = machine.predict_runtime(job, niter)
            cache[key] = value
        return value
    return runtime

def make_commtime_fn(computation, system, niter=3):
    """
    Compute communication times

    Returns Dict mapping (job, worker, worker) to communication time

    This dict is actually a function disguised as a dict with caching enabled
    """
    jobs = computation.jobs
    machines = system.machines
    N = system.comm
    cache = dict()
    def commtime(job1, job2, m1, m2):
        if m1==m2:
            return 0
        key = (job1, job2, machine_type_id(m1), machine_type_id(m2))
        if key in cache:
            total_time = cache[key]
        else:
            total_time = 0
            if job2 is not None:
                variables = set(job1.outputs).intersection(set(job2.inputs))
            else:
                variables = job1.outputs
            for var in variables:
                total_time += N.predict_transfer_time(m1, m2, var)
            cache[key] = total_time
        return total_time

    return commtime
def make_commtime_fn_tompkins(*args, **kwargs):
    """
    See make_commtime_fn

    This edits the arguments of the returned function to remomve the second
    input
    """
    commtime = make_commtime_fn(*args, **kwargs)
    return lambda j1, m1, m2 : commtime(j1, None, m1, m2)

class FunctionDict(object):
    def __init__(self, f):
        self._f = f
    def __getitem__(self, key):
        return self._f(*key)
