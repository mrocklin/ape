from computation import Job, Variable, Node, Computation
import theano
import numpy as np
from theano_to_milp import intermediate_shapes

job_names = {}

class TheanoJob(Job):
    def __init__(self, apply):
        assert isinstance(apply, theano.Apply)
        self._apply = apply

    def type_check(self):
        super(TheanoJob, self).type_check()
        assert all(isinstance(var, TheanoVariable) for var in self.inputs)
        assert all(isinstance(var, TheanoVariable) for var in self.outputs)
        assert isinstance(self.op, theano.Op)

    @property
    def inputs(self):
        return [TheanoArrayVariable(var) for var in self._apply.inputs]
    @property
    def outputs(self):
        return [TheanoArrayVariable(var) for var in self._apply.outputs]
    @property
    def op(self):
        return self._apply.op

    @property
    def name(self):
        return "%s_%d"%(str(self.op), abs(hash(self)))

    def info(self):
        return self._apply

    def function(self, additional_tags=None, gpu=False):
        mode = theano.compile.mode.get_default_mode()

        inputs = self.inputs
        output = self.outputs[0]
        inputs = [Var.var for Var in inputs]
        output = output.var
        assert len(self.outputs) == 1 , "Multiple output assumption fails"

        if gpu:
            inputs, cpu_inputs = zip(*map(cpu_var_to_gpu_var, inputs))
            output = self.op(*cpu_inputs)
            output = theano.sandbox.cuda.basic_ops.gpu_from_host(output)
            output = theano.Out(output, borrow=True)
        else:
            mode = mode.excluding('gpu')

        if additional_tags:
            mode = mode.including(additional_tags)

        return theano.function(inputs, output, mode=mode, name='test')

    def compiler(self):
        return TheanoJob(apply_clone(self._apply))

class StartJob(Job):
    def __init__(self, var):
        self._output = var

    def info(self):
        return self._output, self.__class__

    @property
    def name(self):
        return "Start_%s"%str(self._output)

    @property
    def outputs(self):
        return [self._output]
    @property
    def inputs(self):
        return []
    def __str__(self):
        return self.name
    def compiler(self):
        def f():
            return 0
        return f

class EndJob(Job):
    def __init__(self, var):
        self._input = var

    def info(self):
        return self._input, self.__class__

    @property
    def name(self):
        return "End_%s"%str(self._input)

    @property
    def outputs(self):
        return []
    @property
    def inputs(self):
        return [self._input]
    def __str__(self):
        return self.name
    def compiler(self):
        def f():
            return 0
        return f

class TheanoVariable(Variable):
    def __init__(self, variable):
        self._variable = variable

    def type_check(self):
        super(TheanoVariable, self).type_check()
        assert isinstance(self._variable, theano.Variable)

    def info(self):
        return self._variable

    @property
    def name(self):
        varname = self._variable.name or "var"
        return "%s_%d"%(varname, abs(hash(self)))

    @property
    def from_job(self):
        if not self._variable.owner:
            return StartJob(self)
        return TheanoJob(self._variable.owner)
    @property
    def to_jobs(self):
        if all(isinstance(apply, str) for apply, idx in self._variable.clients):
            return [EndJob(self)]
        assert not any(isinstance(apply, str)
                for apply, idx in self._variable.clients)

        return [TheanoJob(client) for client, index in self._variable.clients]

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def var(self):
        return self._variable

class TheanoArrayVariable(TheanoVariable):
    known_shapes = {}

    def __init__(self, variable, shape=None):
        self._variable = variable
        self._shape = shape
        if not shape and variable.name in TheanoArrayVariable.known_shapes:
            self._shape = TheanoArrayVariable.known_shapes[variable.name]

    def type_check(self):
        super(TheanoArrayVariable, self).type_check()
        assert isinstance(self._variable, theano.tensor.TensorVariable)

    def get_shape(self):
        if not self._shape:
            raise ValueError("Have not yet specified shape for %s"%(str(self)))
        return self._shape
    def set_shape(self):
        self._shape = shape

    shape = property(get_shape, set_shape)
    @property
    def dtype(self):
        return self._variable.dtype

class TheanoComputation(Computation):
    def __init__(self, f, shapes):
        self.f = f
        self.known_shapes = shapes

    def type_check(self):
        assert isinstance(f, theano.function)
        assert isinstance(shapes, dict)

    @property
    def env(self):
        return self.f.maker.env

    @property
    def inputs(self):
        return [TheanoArrayVariable(var, self.known_shapes[var.name])
                for var in self.env.inputs]
    @property
    def outputs(self):
        return [TheanoArrayVariable(var, self.known_shapes[var.name])
                for var in self.env.outputs]

    @property
    def start_jobs(self):
        return map(StartJob, self.inputs)

def cpu_var_to_gpu_var(x):
    from theano.sandbox import cuda
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = 'gpu_%s'%x.name
    name = None
    gpu_var = cuda.CudaNdarrayVariable(type=type, name=name)
    cpu_var = cuda.host_from_gpu(gpu_var)
    return gpu_var, cpu_var
    return cuda.host_from_gpu(cuda.CudaNdarrayVariable(type=type, name=name))

def all_jobs(job):
    jobs = set([])
    def add(j):
        if not j or j in jobs:
            return
        jobs.add(j)
        for k in j.children+j.parents:
            add(k)
    add(job)
    return jobs

def apply_clone(ap):
    """
    Takes in an apply node in some larger Env context.
    Returns the same apply/variables outside of the context
    """
    inputs = [inp.clone() for inp in ap.inputs]
    output = ap.op(*inputs)
    ap_new = output.owner
    return ap_new

