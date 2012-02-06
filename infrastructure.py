from graph import Node
from computation import Variable, Job
import time

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

