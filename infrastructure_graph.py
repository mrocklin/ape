from graph import Node, is_ordered_iterator
from computation_graph import TheanoVariable, TheanoArrayVariable, TheanoJob
from graph import Variable, Job
import time

def importall(view):
    view.execute('import numpy as np')
    view.execute('from computation_graph import *')
def apply_clone(ap):
    """
    Takes in an apply node in some larger Env context.
    Returns the same apply/variables outside of the context
    """
    inputs = [inp.clone() for inp in ap.inputs]
    output = ap.op(inputs)
    ap_new = output.owner
    return ap_new



class Worker(Node):
    """
    run(job)
    predict_runtime(job)
    can_compute(job)
    __contains__(job or var)

    compile(job)
    store_random_instance_of(var)
    """
    def type_check(self):
        assert is_ordered_iterator(self.in_wires)
        assert is_ordered_iterator(self.out_wires)
        assert all(isinstance(w, Wire) for w in self.in_wires)
        assert all(isinstance(w, Wire) for w in self.out_wires)

    def __str__(self):
        return "Worker: "+str(self.name)

    def run(self, job):
        for var in job.inputs:
            assert var in self, "Variable not present on Worker"
        assert job in self, "Job not yet compiled on Worker"

        self._run(job)

        for var in job.outputs:
            assert var in self, "Output variable not produced"

    def predict_runtime(self, job, niter=10):
        for var in job.inputs:
            if not var in self:
                self.instantiate_random_variable(var)
        if not job in self:
            self.compile(job)

        starttime = time.time()
        for i in xrange(niter):
            self._run(job)
        endtime = time.time()

        return (endtime - starttime) / niter

    def can_compute(self, job):
        try:
            self.compile(job)
            return True
        except:
            return False

    def compile(self, job):
        raise NotImplementedError()

    def has_variable(self, var):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    def __contains__(self, x):
        if isinstance(x, Variable):
            return self.has_variable(x)
        if isinstance(x, Job):
            return self.has_function(x)

    def local_name(self, x):
        return x.name

def has_gpu(remote):
    raise NotImplementedError()
class PUWorker(Worker):
    """
    This class contains shared code between CPU and GPU workers
    """
    def has(self, x):
        self.do('result = "%s" in globals()'%self.local_name(x))
        return self.rc['result']

    def has_variable(self, var):
        assert isinstance(var, Variable)
        return self.has(var)

    def has_function(self, job):
        assert isinstance(job, Job)
        return self.has(job)

    def _compile_job(self, job, gpu=None):
        assert gpu is not None
        name = self.local_name(job)
        ap_new = apply_clone(job._apply)
        self.rc['apply_%s'%job.name] = ap_new
        self.do('job_%s = TheanoJob(apply_%s)'%(name, name))
        return self.do('%s = job_%s.function(gpu=%s)'%(name, name, str(gpu)))

    def _run(self, job):
        name = self.local_name
        outputs = ','.join([name(o) for o in job.outputs])
        inputs = ','.join([name(i) for i in job.inputs])

        return self.do('%s = %s([%s])'%(outputs, name(job), inputs))

class GPUWorker(PUWorker):

    def __init__(self, host):
        self.host = host
        assert has_gpu(host), "Can not create a GPUWorker on %s"%host
        self.rc = host.rc

    def compile(self, job):
        return self._compile_job(job, gpu=True)

    def instantiate_random_variable(self, var):
        var_previously_on_host = var in self.host

        if not var_previously_on_host:
            self.host.instantiate_random_variable(var) # create on host

        self.do('%s = togpu_data(%s)'%(
            self.local_name(var), self.host.local_name(var))) # transfer to gpu

        if not var_previously_on_host:
            self.do('del %s'%self.host.local_name(var)) # delete host copy

    def local_name(self, var):
        return "gpu_"+var.name

    @property
    def name(self):
        return self.host.name+"_gpu"

class CPUWorker(PUWorker):
    def __init__(self, remote):
        self.rc = remote

    def do(self, cmd):
        return self.rc.execute(cmd)

    def type_check(self):
        super(CPU_Worker, self).type_check()
        assert isinstance(self.rc, IPython.parallel.client.view.DirectView)

    def get_mpi_rank(self):
        raise NotImplementedError()

    def get_0mq_id(self):
        return self.rc.targets

    @property
    def name(self):
        return str(self.rc.targets)

    def compile(self, job):
        return self._compile_job(job, gpu=False)

    def instantiate_random_variable(self, var):
        assert hasattr(var, 'shape') and hasattr(var, 'dtype')
        name = self.local_name(var)
        self.rc['%s_shape'%name] = var.shape
        self.rc['%s_dtype'%name] = var.dtype

        return self.do('%s = np.random.random(%s_shape).astype(%s_dtype)'%(
            name, name, name))

class Wire(Node):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def send(self, var):
        raise NotImplementedError()

class CPUWireCPU(Wire):
    pass
class MPIWire(CPUWireCPU):
    pass
class ZMQWire(CPUWireCPU):
    pass

class CPUWireGPU(Wire):
    pass
class GPUWireCPU(Wire):
    pass


class CommNetwork(object):
    """
    wires :: Machine, Machine -> Wire
    transfer :: Machine, Machine, Variable -> se
    predict_transfer_time :: Machine, Machine, Variable -> Time
    """
    def __init__(self, *args):
        raise NotImplementedError()

    def __getitem__(self, key):
        A,B = key
        return self._wires[A, B]

    def transfer(self, A, B, V):
        wire = self[A,B]
        assert V in A, "Sending machine does not have variable %s"%V
        wire.send(V)

    def predict_transfer_time(self, A, B, V, niter=10):
        A.store_random_instance_of(V)
        wire = self[A,B]

        starttime = time.time()
        for i in xrange(niter):
            wire.send(V) # make sure we block here

        endtime = time.time()

        return (endtime - starttime) / niter

from IPython.parallel import Client
rc = Client()
view = rc[:]
view.execute('import numpy as np')
view.execute('from computation_graph import TheanoJob, TheanoVariable')
A,B = rc[0], rc[1]
C = CPUWorker(A)
D = CPUWorker(B)
