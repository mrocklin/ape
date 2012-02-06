import itertools
from collections import defaultdict

def is_ordered_iterator(x):
    return isinstance(x, (list, tuple))

class Node(object):
    node_color='red'
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(self.info())
    def __eq__(self, other):
        return self.info() == other.info()

    def to_network(self, graph=None):
        import networkx as nx
        if not graph:
            graph = nx.DiGraph()
            graph.add_node(self)
        for parent in self.parents:
            if parent in graph:
                continue
            graph.add_node(parent)
            graph.add_edge(self, parent)
            graph = parent.to_network(graph)
        return graph

    def draw(self):
        import networkx as nx
        G = self.to_network()
        nx.draw_networkx(G, node_color=[n.node_color for n in G.nodes()])

class Job(Node):
    node_color='red'
    @property
    def inputs(self):
        raise NotImplementedError()
    @property
    def outputs(self):
        raise NotImplementedError()
    @property
    def op(self):
        raise NotImplementedError()

    def type_check(self):
        assert is_ordered_iterator(self.inputs)
        assert is_ordered_iterator(self.outputs)
        assert len(self.inputs)>0 and len(self.outputs)>0

    def info(self):
        return self.inputs, self.outputs, self.op

    def __str__(self):
        return str(self.op)

    def vars_sent_to(self, other):
        return set(self.outputs).intersection(set(other.inputs))

    def executed_directly_before(self, other):
        return len(self.vars_sent_to(other)) != 0

    def precedes(self, other):
        return self.executed_directly_before(other)

    def executed_directly_after(self, other):
        return other.executed_directly_before(self)

    def jobs_executed_directly_after(self):
        return list(itertools.chain(*[var.to_jobs for var in self.outputs]))
    def jobs_executed_directly_before(self):
        return [var.from_job for var in self.inputs]

    @property
    def children(self):
        return self.jobs_executed_directly_after()
    @property
    def parents(self):
        return self.jobs_executed_directly_before()

    def print_child_tree(self, indent=''):
        print(indent + str(self))
        for child in self.children:
            child.print_child_tree(indent+'  ')
    def print_parent_tree(self, indent=''):
        print(indent + str(self))
        for parent in self.parents:
            parent.print_parent_tree(indent+'  ')

class Variable(Node):
    node_color = 'blue'
    @property
    def name(self):
        raise NotImplementedError()
    @property
    def from_job(self):
        raise NotImplementedError()
    @property
    def to_jobs(self):
        raise NotImplementedError()
    @property
    def shape(self):
        raise NotImplementedError()

    def __str__(self):
        return str(self._variable)

    def type_check(self):
        if self.from_job:
            assert isinstance(self.from_job, Job)
        #assert is_ordered_iterator(self.from_job)
        assert is_ordered_iterator(self.to_jobs)
        assert all(isinstance(x, Job) for x in self.to_jobs)

    def info(self):
        return self.from_job, self.to_jobs, self.name, self.shape

    @property
    def is_input_variable(self):
        return self.from_job is None
        #return len(self.from_job)==0
    @property
    def is_output_variable(self):
        return len(self.to_jobs)==0

    @property
    def children(self):
        return self.to_jobs
    @property
    def parents(self):
        if self.from_job is not None:
            return [self.from_job]
        else:
            return []

class Computation(object):

    @property
    def jobs(self):
        s = set()
        def add_descendents(job):
            if job in s:
                return
            s.add(job)
            for child in job.children:
                add_descendents(child)

        for job in self.start_jobs:
            add_descendents(job)
        return s

    @property
    def variables(self):
        s = set()
        for job in self.jobs:
            for var in job.inputs+job.outputs:
                s.add(var)
        return s

    def precedence(self):
        P = defaultdict(lambda:0)
        visited = set()
        def add_precedence_info(j):
            if j in visited:
                return
            visited.add(job)
            for child in j.children:
                P[child,j] = 1 # child comes before j
                add_precedence_info(child)

        # Start recursion from each of the inputs
        for job in self.start_jobs:
                add_precedence_info(job)
        return P

