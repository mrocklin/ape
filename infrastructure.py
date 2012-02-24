from graph import Node
from computation import Variable, Job
import time

a_big_number = 1e9

class Worker(Node):
    """
    run(job)
    predict_runtime(job)
    can_compute(job)
    __contains__(job or var)

    compile(job)
    store_random_instance_of(var)
    """
    def detete(self, var):
        raise NotImplementedError()

    def type_check(self):
        assert is_ordered_iterator(self.in_wires)
        assert is_ordered_iterator(self.out_wires)
        assert all(isinstance(w, Wire) for w in self.in_wires)
        assert all(isinstance(w, Wire) for w in self.out_wires)

    def __str__(self):
        return "Worker: "+str(self.name)

    def run(self, job):
        for var in job.inputs:
            assert var in self, "Input Variable %s not present on Worker"%var
        assert job in self, "Job %s not yet compiled on Worker"%job

        res = self._run(job)
        assert res.result is None # check for raised errors

        for var in job.outputs:
            assert var in self, "Output variable not produced"

    def predict_runtime(self, job, niter=3):
        for var in job.inputs:
        #    if not var in self:
            self.instantiate_random_variable(var)
        if not job in self:
            self.compile(job, block=True)

        starttime = time.time()
        for i in xrange(niter):
            self._run(job).wait()
        endtime = time.time()

        for var in job.inputs+job.outputs:
            res = self.delete(var)
        res.result
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
        return self.__class__.cls_local_name(x)

    _name_prefix = "AAA"
    _name_dict = {}
    _count = 0
    @classmethod
    def cls_local_name(cls, var):
        if   isinstance(var, Variable):         var_prefix = 'var'
        elif isinstance(var, Job):              var_prefix = 'job'
        else:                                   var_prefix = ''

        if var in cls._name_dict:
            return cls._name_dict[var]
        else:
            name = "%s_%s_%d"%(cls._name_prefix, var_prefix, cls._count)
            cls._count += 1
            cls._name_dict[var] = name
            return name


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

    def __str__(self):
        return "%s ---- %s"%(str(self.A), str(self.B))

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
        """
        Send a Variable from Worker A to Worker B
        """
        path = self.path(A,B)
        assert V in A, "Sending machine does not have variable %s"%V
        for wire in path:
            wire.transmit(V)

    def transfer_code(self, A, B, V):
        """
        Produces code to execute a transfer of variable V from A to B

        returns a dict mapping machines to code they should run

        See Also
        --------
        receive_code
        """
        path = self.path(A,B)
        assert len(path)<=1, "Not currently set up for asynchronous routing,  need to set up blocking calls"
        code = {}
        for wire in path:
            acode, bcode = wire.transmit_code(V)
            if wire.A not in code: code[wire.A] = []
            if wire.B not in code: code[wire.B] = []
            code[wire.A] = acode
            code[wire.B] = bcode
        return code
    def receive_wait_code(self, A, B, V):
        """
        Produces code to wait for completion of variable transfer V from A to B

        returns a dict mapping machines to code they should run

        See Also
        --------
        transfer_code
        """
        path = self.path(A,B)
        assert len(path)<=1, "Not currently set up for asynchronous routing,  need to set up blocking calls"
        code = {}
        for wire in path:
            sendacode, recvbcode = wire.waiting_code(V)
            code[wire.B] = recvbcode
        return code

    def path(self, A, B):
        from theano_infrastructure import GPUWorker # breaking dependencies!!
        if A==B:        return []
        try:            return [self[A,B]]
        except:         pass
        # no trivial solution - now we deal with routing

        # only a simple algorithm so far to deal with GPUs
        if isinstance(B, GPUWorker):
            return [self[A,B.host], self[B.host, B]]
        if isinstance(A, GPUWorker):
            return [self[A,A.host], self[A.host, B]]

        raise KeyError("Unable to find path from %s to %s"%(str(A), str(B)))

    def predict_transfer_time(self, A, B, V, niter=3):
        try:                path = self.path(A,B)
        except KeyError:    return a_big_number

        A.instantiate_random_variable(V)

        starttime = time.time()
        for i in xrange(niter):
            for wire in path:
                wire.transmit(V) # make sure we block here

        endtime = time.time()

        return (endtime - starttime) / niter

class ComputationalSystem(object):
    def __init__(self, machines, comm):
        self.machines = machines
        self.comm = comm

