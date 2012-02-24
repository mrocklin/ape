from computation import Job, Variable, Node, Computation
import theano
import numpy as np
from theano_util import intermediate_shapes

job_names = {}

class TheanoJob(Job):
    def __init__(self, apply, computation):
        assert isinstance(apply, theano.Apply)
        self._apply = apply
        self.c = computation

    def type_check(self):
        super(TheanoJob, self).type_check()
        assert all(isinstance(var, TheanoVariable) for var in self.inputs)
        assert all(isinstance(var, TheanoVariable) for var in self.outputs)
        assert isinstance(self.op, theano.Op)
        assert isinstance(self.c, TheanoComputation)

    @property
    def inputs(self):
        return [TheanoArrayVariable(var, self.c) for var in self._apply.inputs
                if not isinstance(var, theano.Constant)]
    @property
    def outputs(self):
        return [TheanoArrayVariable(var, self.c) for var in self._apply.outputs]
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
        return TheanoJob(apply_clone(self._apply), None)

    def __str__(self):
        return str(self.op)+suffix(self)

def suffix(x):
    """
    Many different jobs will have the same op and thus the same name. Lets add
    suffixes to get rid of this problem
    """
    if x in suffix._cache:
        return suffix._cache[x]
    s = '_%d'%(suffix._count)
    suffix._count += 1
    suffix._cache[x] = s
    return s
suffix._cache = dict()
suffix._count = 0
name = lambda x:x

class simplecompiler(object):
    def __init__(self):
        pass
    def function(self, *args, **kwargs):
        return lambda :0

class StartOrEndJob(Job):
    def __init__(self, var):
        self._var = var

    def info(self):
        return self._var, self.__class__

    def __str__(self):
        return self.name

    def function(*args, **kwargs):
        return lambda : 0

    def compiler(self):
        return simplecompiler()

class StartJob(StartOrEndJob):

    @property
    def name(self):
        return "Start_%s"%str(self._var)
    @property
    def outputs(self):
        return [self._var]
    @property
    def inputs(self):
        return []

class EndJob(StartOrEndJob):

    @property
    def name(self):
        return "End_%s"%str(self._var)
    @property
    def outputs(self):
        return []
    @property
    def inputs(self):
        return [self._var]

class TheanoVariable(Variable):
    def __init__(self, variable, computation):
        self._variable = variable
        self.c = computation

    def type_check(self):
        super(TheanoVariable, self).type_check()
        assert isinstance(self._variable, theano.Variable)
        assert isinstance(self.c, TheanoComputation)

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
        return TheanoJob(self._variable.owner, self.c)
    @property
    def to_jobs(self):
        if all(isinstance(apply, str) for apply, idx in self._variable.clients):
            return [EndJob(self)]
        assert not any(isinstance(apply, str)
                for apply, idx in self._variable.clients)

        return [TheanoJob(client, self.c)
                for client, index in self._variable.clients]

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def var(self):
        return self._variable

class TheanoArrayVariable(TheanoVariable):
    def __init__(self, variable, computation):
        self._variable = variable
        try:        self._shape = computation.known_shapes[variable.name]
        except:     self._shape = None
        self.c = computation

    def type_check(self):
        super(TheanoArrayVariable, self).type_check()
        assert isinstance(self._variable, theano.tensor.TensorVariable)

    def get_shape(self):
        if self._shape is None:
            raise ValueError("Have not yet specified shape for %s"%(str(self)))
        return self._shape
    def set_shape(self):
        self._shape = shape

    shape = property(get_shape, set_shape)
    @property
    def dtype(self):
        return self._variable.dtype

def give_names_to_function(f):
    for var in f.maker.env.variables:
        if not var.name:
            var.name = "unnamed_var_%d"%give_names_to_function.count
            give_names_to_function.count+=1
    return f
give_names_to_function.count = 0


class TheanoComputation(Computation):
    def __init__(self, f, shapes):
        self.f = f
        give_names_to_function(f)
        self.known_shapes = self.compute_known_shapes(shapes)


    def type_check(self):
        assert isinstance(self.f, theano.function)
        assert isinstance(self.known_shapes, dict)

    @property
    def env(self):
        return self.f.maker.env

    @property
    def inputs(self):
        return [TheanoArrayVariable(var, self)
                for var in self.env.inputs]
    @property
    def outputs(self):
        return [TheanoArrayVariable(var, self)
                for var in self.env.outputs]

    #@property
    #def jobs(self):
    #    return set(TheanoJob(node, self) for node in self.env.nodes)

    @property
    def varibles(self):
        return set(TheanoArrayVariable(var, self) for var in self.env.variables)

    @property
    def start_jobs(self):
        return map(StartJob, self.inputs)
    @property
    def end_jobs(self):
        return map(EndJob, self.outputs)

    def compute_known_shapes(self, inputshapes):
        symbolic_shapes = self.f.maker.env.shape_feature.shape_of
        numeric_shapes = {}
        inputs = self.f.maker.env.inputs

        # convert all symbolic input shapes to numeric equivalents
        sym_to_num = {}
        for input, num_shape in zip(inputs, inputshapes):
            sym_shape = symbolic_shapes[input]
            for sym, num in zip(sym_shape, num_shape):
                sym_to_num[sym] = num

        for var, sym_shape in symbolic_shapes.items():
            num_shape = tuple([sym_to_num[sym] for sym in sym_shape])
            numeric_shapes[var.name] = num_shape

        return numeric_shapes

        variables = set()
        def get_variables(v):
            if v in variables:
                return
            variables.add(v)
            for client, index in v.clients:
                if isinstance(client, str): return
                for var in client.outputs:
                    get_variables(var)

        inputs = self.env.inputs
        for var in inputs:
            get_variables(var)

        shape_outputs = [var.shape for var in variables]
        compute_shapes = theano.function(inputs, shape_outputs)

        def tuplify_shape(shape):
            #if len(shape)==0:   return (1,)
            #else:               return tuple(shape)
            return tuple(shape)

        numeric_inputs = [np.ones(shape).astype(np.float32)
                for shape in inputshapes]

        shapes = compute_shapes(*numeric_inputs)
        # known_shapes = dict(zip(variables, map(tuplify_shape, shapes)))
        known_shapes = dict()
        for var, shape in zip(variables, shapes):
            known_shapes[var.name] = tuplify_shape(shape)

        return known_shapes

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
    if isinstance(output, list): output = output[0]
    ap_new = output.owner
    return ap_new

