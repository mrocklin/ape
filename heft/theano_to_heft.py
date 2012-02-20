from heft import schedule

big_number = 1e9
def theano_heft_schedule(computation, system, startmachine, **kwargs):
    machines = system.machines
    network = system.comm

    # jobs = map(name, computation.jobs)
    jobs = computation.jobs
    workers = machines

    def runtime(job, worker):
        terminal_jobs = computation.start_jobs + computation.end_jobs
        if job in terminal_jobs and worker!=startmachine:
            return big_number
        return worker.predict_runtime(job)

    def commtime(job1, job2, worker1, worker2):
        variables = set(job1.outputs).intersection(set(job2.inputs))

        total_time = 0
        for v in variables:
            total_time += network.predict_transfer_time(
                    worker1, worker2, v, **kwargs)

        return total_time

    return schedule(jobs, workers, computation.start_jobs,
                    computation.end_jobs, runtime, commtime)

