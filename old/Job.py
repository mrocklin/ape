InputVariable = 'an input variable'
import theano
from util import set_union

class Node(object):
    is_Variable = False
    is_InputVariable = False
    is_OutputVariable = False
    is_Job = False

    def __new__(cls, data):
        if isinstance(data, theano.gof.graph.Apply):
            return Job(data)
        elif isinstance(data, theano.tensor.basic.TensorVariable):
            if data.owner is None:
                return InputVariable(data)
        elif isinstance(data, str):
            return OutputVariable(data)
        raise NotImplementedError("Confused by input %s"%str(data))

    def __hash__(self):
        return hash(self.data)
    def __eq__(self, other):
        return self.data == other.data

    def __repr__(self):
        return str(self)

    def descendents(self):
        return set_union([chld.descendents() for chld in self.children] +
                [{self}])

    def ancestors(self):
        return set_union([parent.ancestors() for parent in self.parents] +
                [{self}])

    @property
    def children(self):
        if not self.apply:
            return set()
        return {Job(inp.owner) if inp.owner
                else InputVariable(inp)
                for inp in self.apply.inputs}

class Variable(Node):
    is_Variable = True
    def print_tree(self, indent=''):
        print(indent + str(self))

    @property
    def apply(self):
        return self.variable.owner

class OutputVariable(Variable):
    is_OutputVariable = True
    def __new__(cls, name, idx, child):
        self = object.__new__(OutputVariable)
        self.name = name
        self.idx = idx
        self.child = child
        return self

    @property
    def data(self):
        return self.name

    def __hash__(self):
        return hash(self.name, self.idx, self.child)
    def __eq__(self, other):
        return type(self) == type(other) and (
                self.name, self.idx, self.child ==
                other.name, other.idx,other.child )

    def __str__(self):
        return "OutputVariable: %s"%self.variable

    @property
    def parents(self):
        return set()

    @property
    def children(self):
        return {self.child}

class InputVariable(Variable):
    is_InputVariable = True
    def __new__(cls, data):
        self = object.__new__(InputVariable)
        self.variable = data
        return self

    @property
    def data(self):
        return self.variable

    def __str__(self):
        return "InputVariable: %s"%self.variable

    @property
    def children(self):
        return set()

    @property
    def parents(self):
        return {Node(client[0]) for client in self.variable.clients}

class Job(Node):
    is_Job = True
    def __new__(cls, apply_node, start=False):
        self = object.__new__(Job)
        self.apply = apply_node
        return self

    @property
    def data(self):
        return self.apply

    def __str__(self):
        return "Job: %s"%self.apply

    def print_tree(self, indent=''):
        print(indent + str(self))
        for child in self.children:
            child.print_tree(indent+'  ')

    def op(self):
        return self.apply.op

    def _env(self):
        inputs = [inp.clone() for inp in self.apply.inputs]
        output = self.apply.op(inputs)
        env = theano.FunctionGraph(inputs, [output])
        return env

    def function(self, additional_tags=None, gpu=False):
        # inputs = [inp.clone() for inp in self.apply.inputs]
        # output = self.apply.op(inputs)
        # env = theano.FunctionGraph(inputs, [output])
        mode = theano.compile.mode.get_default_mode()

        inputs = self.apply.inputs
        output = self.apply.outputs[0]
        assert len(self.apply.outputs) == 1 , "Multiple output assumption fails"

        if gpu:
            inputs, cpu_inputs = zip(*map(cpu_var_to_gpu_var, inputs))
            output = self.apply.op(*cpu_inputs)
            output = theano.sandbox.cuda.basic_ops.gpu_from_host(output)
            output = theano.Out(output, borrow=True)
        else:
            mode = mode.excluding('gpu')

        if additional_tags:
            mode = mode.including(additional_tags)

        return theano.function(inputs, output, mode=mode)

    def to_network(self, graph=None):
        import networkx as nx
        if not graph:
            graph = nx.DiGraph()
            graph.add_node(self)
        for child in self.children:
            graph.add_node(child)
            graph.add_edge(self, child)
            graph = child.to_network(graph)
        return graph

    @property
    def parents(self):
        return {Node(client[0]) for output in self.apply.outputs
                                for client in output.clients}

def cpu_var_to_gpu_var(x):
    from theano.sandbox import cuda
    type = cuda.CudaNdarrayType(broadcastable=x.broadcastable)
    name = 'gpu_%s'%x.name
    name = None
    gpu_var = cuda.CudaNdarrayVariable(type=type, name=name)
    cpu_var = cuda.host_from_gpu(gpu_var)
    return gpu_var, cpu_var
    return cuda.host_from_gpu(cuda.CudaNdarrayVariable(type=type, name=name))
