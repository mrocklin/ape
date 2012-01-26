from infrastructure_graph import Worker, Wire
import IPython
from IPython.parallel import Client


class CPU_Worker(Worker):
    def __init__(self, remote):
        self.rc = remote

    def do(self, cmd):
        return self.rc.execute(cmd)

    def type_check(self):
        super(CPU_Worker, self).type_check()
        assert isinstance(self.rc, IPython.parallel.client.view.DirectView)

    def get_mpi_rank(self):
        raise NotImplementedError()

    def get_0mq_id(self):
        return self.rc.targets

    def has_variable(self, var):
        assert isinstance(var, Variable)
        self.do('result = %s in globals()'%var.name)
        return self.rc['result']

    def has_function(self, job):
        assert isinstance(job, Job)
        self.do('result = %s in globals()'%job.name)
        return self.rc['result']

    def compile_job(self, job):
        ap = job._apply
        inputs = [inp.clone() for inp in ap.inputs]
        output = ap.op(inputs)
        ap_new = output.owner
        self.rc['apply_%s'%job.name] = ap_new
        self.do('job_%s = TheanoJob(apply_%s)'%(job.name, job.name))
        return self.do('%s = job_%s.function(gpu=False)'%(job.name, job.name))

    def instantiate_random_variable(self, var):
        assert hasattr(var, 'shape') and hasattr(var, 'dtype')
        self.rc['%s_shape'] = var.shape
        self.rc['%s_dtype'] = var.dtype

        return self.do('%s = np.random.random(%s_shape).astype(%s_dtype)'%(
            var.name, var.name, var.name))

    def _run_job(self, job):
        outputs = ','.join([o.name for o in job.outputs])
        inputs = ','.join([i.name for i in job.inputs])

        return self.do('%s = %s(%s)'%(outputs, job.name, inputs))




rc = Client()
view = rc[:]
view.execute('import numpy as np')
view.execute('from numeric_graph import TheanoJob, TheanoVariable')
A,B,C,D = rc[0], rc[1], rc[2], rc[3]
a = CPU_Worker(A)
