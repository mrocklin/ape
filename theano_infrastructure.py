from graph import Node
from util import is_ordered_iterator, host_name
from theano_computation import (TheanoVariable, TheanoArrayVariable, TheanoJob,
        Variable, Job)
from infrastructure import Worker, Wire
import time
import theano
import numpy as np

imports = [ 'import numpy as np',
            'from theano_computation import *',
            'from theano_infrastructure import togpu_data, tocpu_data',
            'import theano',
            'from mpi4py import MPI']
###########
# Workers #
###########

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

    def _compile(self, job, gpu=None, block=False):
        assert gpu is not None
        fnname = self.local_name(job)
        compilername = 'compiler_%s'%fnname

        # Push compilation object (likely a TheanoJob clone)
        res = self.rc.push({compilername: job.compiler()})
        res.wait(); assert res.successful()

        # Tell remote to compile the function locally
        res = self.do('%s = %s.function(gpu=%s)'%
                (fnname, compilername, str(gpu)))
        res.wait(); assert res.successful()

        # Sometimes names are lost in the cloning process. Ensure its ok
        res = self.do('%s.name = %s.name if hasattr(%s, "name") else "%s"'%(
            fnname, fnname, fnname, fnname))
        res.wait(); assert res.successful()
        if block:
            res = res.result
        return res

    def _run_code(self, job):
        if not job.outputs: return ''

        name = self.local_name
        outputs = ', '.join([name(o) for o in job.outputs])
        inputs = ', '.join([name(i) for i in job.inputs])

        return '%s = %s(%s)'%(outputs, name(job), inputs)
    def _run(self, job):
        return self.do(self._run_code(job))

    def __getitem__(self, key):
        if isinstance(key, (Variable, Job)):
            key = self.local_name(key)
        return self.rc[key]

    def info(self):
        return (self.rc, self.__class__)

    def delete(self, var):
        return self.do(self.delete_code(var))
    def delete_code(self, var):
        return 'del %s'%self.local_name(var)

    def get_hostname(self):
        return self.rc.apply_sync(host_name)

class GPUWorker(PUWorker):
    _name_prefix = 'gpu'
    _name_dict = {}

    def __init__(self, host):
        self.host = host
        assert has_gpu(host), "Can not create a GPUWorker on %s"%str(host)
        self.rc = host.rc

    def compile(self, job, block=False):
        return self._compile(job, gpu=True, block=block)

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
        res = self.do(self.instantiate_empty_variable_code(var))
        return res

    def instantiate_empty_variable_code(self, var):
        assert var.dtype in (np.float32, 'float32')
        name = self.local_name(var)
        code = '%s = theano.sandbox.cuda.CudaNdarray.zeros(%s);'%(
                name, str(var.shape))
        return code

    @property
    def name(self):
        return self.host.name+"_gpu"

class CPUWorker(PUWorker):
    _name_prefix = 'cpu'
    _name_dict = {}

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

    def compile(self, job, block=False):
        return self._compile(job, gpu=False, block=block)

    def instantiate_random_variable(self, var):
        assert hasattr(var, 'shape') and hasattr(var, 'dtype')
        name = self.local_name(var)
        self.rc['%s_shape'%name] = var.shape
        self.rc['%s_dtype'%name] = var.dtype

        return self.do('%s = np.random.random(%s_shape).astype(%s_dtype)'%(
            name, name, name))

    def instantiate_empty_variable(self, var):
        assert hasattr(var, 'shape') and hasattr(var, 'dtype')
        return self.do(self.instantiate_empty_variable_code(var)).result

    def instantiate_empty_variable_code(self, var):
        name = self.local_name(var)
        code = '%s = np.empty(%s, dtype="%s");'%(
                name, str(var.shape), str(var.dtype))
        return code

def importall(view):
    for import_line in imports:
        view.execute(import_line)

def has_gpu(remote):
    def device():
        import theano
        return theano.config.device
    return remote.rc.apply_sync(device) == 'gpu'

#########
# Wires #
#########

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
        acode, bcode = self.transmit_code(var)
        a = self.A.do(acode)
        b = self.B.do(bcode)

        return a,b

    def transmit_code(self, var, tag=None):
        if not tag:
            tag = hash(var) % 2**16
        # code for A
        varname = self.A.local_name(var)
        acode = 'mpi_send_%s = MPI.COMM_WORLD.Send(%s, dest=%d, tag=%d);'%(
                     varname, varname, self.b_rank, tag)

        bcode = self.B.instantiate_empty_variable_code(var)

        varname = self.B.local_name(var)
        bcode += 'mpi_recv_%s = MPI.COMM_WORLD.Recv(%s, source=%d, tag=%d);'%(
                     varname, varname, self.a_rank, tag)
        return acode, bcode

    def waiting_code(self, var, tag=None):
        if not tag:
            tag = hash(var) % 2**16
        # code for A
        acode = 'mpi_send_%s.wait()'%self.A.local_name(var)
        bcode = 'mpi_recv_%s.wait()'%self.B.local_name(var)

        return acode, bcode


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

class CPUWireGPU(CGPUWire):
    def __init__(self, A, B):
        self.cpu = self.A = A
        self.gpu = self.B = B
        self.type_check()
    def transmit(self, var):
        cpuname = self.cpu.local_name(var)
        gpuname = self.gpu.local_name(var)
        return self.cpu.do('%s = togpu_data(%s)'%(gpuname, cpuname))


class GPUWireCPU(CGPUWire):
    def __init__(self, A, B):
        self.gpu = self.A = A
        self.cpu = self.B = B
        self.type_check()
    def transmit(self, var):
        cpuname = self.cpu.local_name(var)
        gpuname = self.gpu.local_name(var)
        return self.cpu.do('%s = tocpu_data(%s)'%(cpuname, gpuname))


def togpu_data(x, copy=True):
    if isinstance(x, np.ndarray):
        return theano.sandbox.cuda.shared_constructor(x).get_value(
                borrow=True, return_internal_type=True)
    if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        if copy:
            return x.copy()
        else:
            return x
    if isinstance(x, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
        return xg.get_value(return_internal_type=True, borrow=copy)
    assert False

def tocpu_data(x, copy=True):
    if isinstance(x, theano.sandbox.cuda.CudaNdarray):
        return np.asarray(x)
    if isinstance(x, np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    if isinstance(x, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
        return x.get_value(return_internal_type=False)
    assert False

#from IPython.parallel import Client
#rc = Client()
#view = rc[:]
#importall(view)
#A,B = rc[0], rc[1]
#C = CPUWorker(A)
#D = CPUWorker(B)
