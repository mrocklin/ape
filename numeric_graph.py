from graph import Job as AbstractJob
from graph import Variable as AbstractVariable
from graph import Node as AbstractNode
import theano
import numpy as np

class Job(AbstractJob):
    def __init__(self, apply):
        assert isinstance(apply, theano.Apply)
        self._apply = apply

    def type_check(self):
        super(Job, self).type_check()
        assert isinstance(self.inputs[0], theano.Variable)
        assert isinstance(self.outputs[0], theano.Variable)
        assert isinstance(self.op, theano.Op)

    @property
    def inputs(self):
        return [Variable(var) for var in self._apply.inputs]
    @property
    def outputs(self):
        return [Variable(var) for var in self._apply.outputs]
    @property
    def op(self):
        return self._apply.op

    def info(self):
        return self._apply

    def function(self, additional_tags=None, gpu=False):
        # inputs = [inp.clone() for inp in self.apply.inputs]
        # output = self.apply.op(inputs)
        # env = theano.Env(inputs, [output])
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

        return theano.function(inputs, output, mode=mode)

class Variable(AbstractVariable):
    def __init__(self, variable):
        self._variable = variable

    def type_check(self):
        super(Variable, self).type_check()
        assert isinstance(self._variable, theano.Variable)

    def info(self):
        return self._variable

    @property
    def name(self):
        return self._variable.name

    @property
    def from_job(self):
        if not self._variable.owner:
            return None
        return Job(self._variable.owner)
    @property
    def to_jobs(self):
        if all(isinstance(apply, str) for apply, idx in self._variable.clients):
            return []
        assert not any(isinstance(apply, str)
                for apply, idx in self._variable.clients)

        return [Job(client) for client, index in self._variable.clients]

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def var(self):
        return self._variable

def cpu_var_to_gpu_var(x):
    from theano.sandbox import cuda
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = 'gpu_%s'%x.name
    name = None
    gpu_var = cuda.CudaNdarrayVariable(type=type, name=name)
    cpu_var = cuda.host_from_gpu(gpu_var)
    return gpu_var, cpu_var
    return cuda.host_from_gpu(cuda.CudaNdarrayVariable(type=type, name=name))
