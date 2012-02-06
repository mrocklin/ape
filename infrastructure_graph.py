from graph import Node, is_ordered_iterator
from computation_graph import TheanoVariable, TheanoArrayVariable, TheanoJob
from graph import Variable, Job
from theano_to_milp import togpu_data, tocpu_data
import time
import theano

def importall(view):
    view.execute('import numpy as np')
    view.execute('from computation_graph import *')
    view.execute('from theano_to_milp import togpu_data, tocpu_data')
    view.execute('import theano')
    view.execute('from mpi4py import MPI')

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
            self._run(job).wait()
        endtime = time.time()

        return (endtime - starttime) / niter

    def can_compute(self, job):
        try:
            res = self.compile(job)
            res = res.result # pull the error from the remote job if one exists
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

    def instantiate_random_variable(self, var):
        raise NotImplementedError()
    def instantiate_empty_variable(self, var):
        raise NotImplementedError()

def has_gpu(remote):
    def device():
        import theano
        return theano.config.device
    return remote.rc.apply_sync(device) == 'gpu'

class PUWorker(Worker):
    """
    This class contains shared code between CPU and GPU workers
    """

    def has(self, x):
        self.do('result = "%s" in globals()'%self.local_name(x))
        return self.rc['result']

    def do(self, cmd):
        return self.rc.execute(cmd)

    def has_variable(self, var):
        assert isinstance(var, Variable)
        return self.has(var)

    def has_function(self, job):
        assert isinstance(job, Job)
        return self.has(job)

    def _compile(self, job, gpu=None):
        assert gpu is not None
        name = self.local_name(job)
        self.rc['job_%s'%name] = job.compiler()
        self.do('%s = job_%s.function(gpu=%s)'%(name, name, str(gpu)))
        res = self.do('%s.name = %s.name if hasattr(%s, "name") else %s'%(
            name, name, name, name))
        return res

    def _run(self, job):
        name = self.local_name
        outputs = ','.join([name(o) for o in job.outputs])
        inputs = ','.join([name(i) for i in job.inputs])

        return self.do('%s = %s(%s)'%(outputs, name(job), inputs))

    def __getitem__(self, key):
        if isinstance(key, (Variable, Job)):
            key = self.local_name(key)
        return self.rc[key]

    def info(self):
        return (self.rc, self.__class__)

class GPUWorker(PUWorker):

    def __init__(self, host):
        self.host = host
        assert has_gpu(host), "Can not create a GPUWorker on %s"%str(host)
        self.rc = host.rc

    def compile(self, job):
        return self._compile(job, gpu=True)

    def instantiate_random_variable(self, var):
        var_previously_on_host = var in self.host

        if not var_previously_on_host:
            self.host.instantiate_random_variable(var) # create on host

        res = self.do('%s = togpu_data(%s)'%(
            self.local_name(var), self.host.local_name(var))) # transfer to gpu

        if not var_previously_on_host:
            self.do('del %s'%self.host.local_name(var)) # delete host copy

        return res

    def instantiate_empty_variable(self, var):
        assert var.dtype in (np.float32, 'float32')
        name = self.local_name(var)
        res = self.do('%s = theano.sandbox.cuda.CudaNdarray.zeros(%s)'%(
                      name,                              str(var.shape)))
        return res

    def local_name(self, var):
        return "gpu_"+var.name

    @property
    def name(self):
        return self.host.name+"_gpu"

class CPUWorker(PUWorker):
    def __init__(self, remote):
        self.rc = remote

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
        return self._compile(job, gpu=False)

    def instantiate_random_variable(self, var):
        assert hasattr(var, 'shape') and hasattr(var, 'dtype')
        name = self.local_name(var)
        self.rc['%s_shape'%name] = var.shape
        self.rc['%s_dtype'%name] = var.dtype

        return self.do('%s = np.random.random(%s_shape).astype(%s_dtype)'%(
            name, name, name))

    def instantiate_empty_variable(self, var):
        assert hasattr(var, 'shape') and hasattr(var, 'dtype')
        name = self.local_name(var)

        return self.do('%s = np.empty(%s, dtype="%s")'%(
            name, str(var.shape), str(var.dtype))).result


class Wire(Node):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def transmit(self, var):
        raise NotImplementedError()

    def info(self):
        return (self.A, self.B, self.__class__)

    def type_check(self):
        assert isinstance(self.A, Worker)
        assert isinstance(self.B, Worker)

class CPUWireCPU(Wire):
    def type_check(self):
        assert isinstance(self.A, CPUWorker)
        assert isinstance(self.B, CPUWorker)

class MPIWire(CPUWireCPU):
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.init_comm()

    def init_comm(self):
        def init_mpi():
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm.Get_rank()

        self._a_rank = self.A.rc.apply_sync(init_mpi)
        self._b_rank = self.B.rc.apply_sync(init_mpi)
        return

    def get_ranks(self):
        def get_rank():
            from mpi4py import MPI
            return MPI.COMM_WORLD.Get_rank()
        return self.A.rc.apply(get_rank), self.B.rc.apply(get_rank)

    @property
    def ranks(self):
        return self._a_rank, self._b_rank

    @property
    def a_rank(self):
        return self.ranks[0]
    @property
    def b_rank(self):
        return self.ranks[1]

    def transmit(self, var):
        assert var in self.A
        self.B.instantiate_empty_variable(var)
        a = self.B.do('MPI.COMM_WORLD.Recv(%s, source=%d, tag=13)'%(
                self.B.local_name(var), self.a_rank))
        b = self.A.do('MPI.COMM_WORLD.Send(%s, dest=%d, tag=13)'%(
                self.A.local_name(var), self.b_rank))

        return a,b
class ZMQWire(CPUWireCPU):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def init_comm(self):
        for W in [self.A, self.B]:
            W.execute('from communicator import EngineCommunicator')
            W.execute('if "com" not in globals(): com = EngineCommunicator()')
        self.peers = {self.A : self.A.rc.apply_async(lambda : com.info),
                      self.B : self.B.rc.apply_async(lambda : com.info)}
        for W in [self.A, self.B]:
            W.rc.apply_sync(lambda pdict: com.connect(pdict), peers)


class CGPUWire(Wire):

    def type_check(self):
        super(CGPUWire, self).type_check()
        assert isinstance(self.cpu, CPUWorker)
        assert isinstance(self.gpu, GPUWorker)
        assert self.gpu.host == self.cpu

    def transmit(self, var):
        cpuname = self.cpu.local_name(var)
        gpuname = self.gpu.local_name(var)
        return self.cpu.do('%s = togpu_data(%s)'%(gpuname, cpuname))

class CPUWireGPU(CGPUWire):
    def __init__(self, A, B):
        self.cpu = self.A = A
        self.gpu = self.B = B
        self.type_check()

class GPUWireCPU(CGPUWire):
    def __init__(self, A, B):
        self.gpu = self.A = A
        self.cpu = self.B = B
        self.type_check()

class CommNetwork(object):
    """
    wires :: Machine, Machine -> Wire
    transfer :: Machine, Machine, Variable -> se
    predict_transfer_time :: Machine, Machine, Variable -> Time
    """
    def __init__(self, wires):
        self._wires = {(w.A, w.B):w for w in wires}

    def __getitem__(self, key):
        A,B = key
        return self._wires[A, B]

    def transfer(self, A, B, V):
        wire = self[A,B]
        assert V in A, "Sending machine does not have variable %s"%V
        wire.transmit(V)

    def predict_transfer_time(self, A, B, V, niter=10):
        if A==B:
            return 0
        try:
            wire = self[A,B]
        except KeyError:
            return -1

        A.instantiate_random_variable(V)

        starttime = time.time()
        for i in xrange(niter):
            wire.transmit(V) # make sure we block here

        endtime = time.time()

        return (endtime - starttime) / niter

class ComputationalNetwork(object):
    def __init__(self, machines, comm):
        self.machines = machines
        self.comm = comm

#from IPython.parallel import Client
#rc = Client()
#view = rc[:]
#importall(view)
#A,B = rc[0], rc[1]
#C = CPUWorker(A)
#D = CPUWorker(B)
