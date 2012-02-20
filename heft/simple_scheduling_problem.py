from heft import *

class j(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return str(self.name)
    __repr__ = __str__



b, m, s, e = [j('begin'), j('mmul'), j('sum'), j('end')]
b.children = [m]; b.parents = []
m.children = [s]; m.parents = [b]
s.children = [e]; s.parents = [m]
e.children = []; e.parents = [s]

runtime = lambda job, worker : 1
commtime = lambda job1, job2, worker1, worker2 : 0 if worker1==worker2 else 1
inputs = [b]
outputs = [e]

schedule([b,m,s,e], ['worker1', 'worker2'], inputs, outputs, runtime, commtime)
