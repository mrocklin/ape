from graph import Node, is_ordered_iterator
import time

class Worker(Node):

    def type_check(self):
        assert is_ordered_iterator(self.in_wires)
        assert is_ordered_iterator(self.out_wires)
        assert all(isinstance(w, Wire) for w in self.in_wires)
        assert all(isinstance(w, Wire) for w in self.out_wires)

    def run_job(self, job):
        for var in job.inputs:
            assert self.has_variable(var), "Variable not present on Worker"
        assert self.has_function(job), "Job not yet compiled on Worker"
        self._run_job(self, job)
        for var in job.outputs:
            assert self.has_variable(var), "Output variable not produced"

    def cost_do_do(self, job, niter=10):
        for var in job.inputs:
            if not self.has_variable(var):
                self.instantiate_random_variable(var)
        if not self.has_function(job):
            self.compile_job(job)

        starttime = time.time()
        for i in xrange(niter):
            self._run_job(job)
        endtime = time.time()

        return (endtime - starttime) / niter


class Wire(Node):
    pass
