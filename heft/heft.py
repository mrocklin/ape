
def cachify(fn):
    """
    Returns a copy of the input function that caches its results
    """
    _cache = dict()
    def new_fn(*args):
        if args in _cache:
            result = _cache[args]
        else:
            result = fn(*args)
            _cache[args] = result
        return result
    docstring = fn.__doc__ or ""
    name = fn.__name__ or ""
    new_fn.__name__, new_fn.__doc__ = name+":cached", docstring+"cached"
    return new_fn

def mean(l):
    return 1.0*sum(l)/len(l)

average = mean
def ranku(job, workers, runtime, commtime):
    """
    Upward rank of a job

    http://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time#Prioritizing_Tasks
    """
    n_i = job
    w_i_average = average([runtime(job, worker) for worker in workers])
    succ_n_i = job.children
    max_child_times = 0 if not job.children else \
        max([
            average([commtime(n_j, n_i, w_a, w_b)
                                        for w_a in workers for w_b in workers])
            + ranku(n_j, workers, runtime, commtime)
        for n_j in job.children])
    return w_i_average + max_child_times

class WorkerSchedule(object):
    """
    A class to represent the schedule of a worker

    Initialize an empty schedule for a worker.

    best_timeslot(job, duration) :  returns start and finish times of when the
                                    job would be scheduled
    insert(job, start, finish)   :  Schedule a job on this worker at the
                                    specifed times
    assert_schedule()            :  Assert that the schedule is valid
    """
    def __init__(self, worker):
        self.worker = worker
        self.task_list = []
    def insert(self, job, start_time, finish_time):
        i = 0
        for i, (j,st,ft) in enumerate(self.task_list):
            if start_time>ft:
                break
        self.task_list.insert(i+1, (job, start_time, finish_time))
        self.assert_schedule()

    def assert_schedule(self):
        for (_,_,ft1),(_,st2,_) in zip(self.task_list[:-1], self.task_list[1:]):
            assert ft1<=st2

    def best_timeslot_after(self, job, ready_time, duration):
        for t1,t2 in zip(self.task_list, self.task_list[1:]+[(0,1e300,0)]):
            _,_,finish1 = t1
            _,start2,_  = t2
            potential_start  = finish1
            potential_finish = finish1+duration
            if potential_finish < start2 and potential_finish > ready_time:
                return potential_start, potential_finish
        start = max(ready_time,
        return start,  start+duration
        potential_start = ready_time
        potential_finish = ready_time+duration


def finish_time(job, duration, worker_schedule):
    start, finish = worker_schedule.best_timeslot(job, duration)
    return finish

def schedule(jobs, workers, inputs, outputs, runtime, commtime, cache=True):
    if cache:
        runtime = cachify(runtime)
        commtime = cachify(commtime)

    def priority(job):
        return ranku(job, workers, runtime, commtime)

    priority_list = [(priority(job), job) for job in jobs]
    priority_list.sort(key = lambda (p, j) : -p) # sort by priority, highest 1st

    worker_schedules = map(WorkerSchedule, workers)

    # Returns when a job would be scheduled on a worker_schedule
    def timings(job, worker_schedule):
        duration = runtime(job, worker_schedule.worker)
        start, finish = worker_schedule.best_timeslot(job, duration)
        return start, finish, worker_schedule

    # Go through the list of jobs starting with highest priority
    for priority, job in priority_list:
        # Compute when this job would finish on each worker
        start_finish_ws = [timings(job, ws) for ws in worker_schedules]
        # Greedily select the best
        start_finish_ws.sort(key = lambda (s,f,ws) : -f) # sort by earliest fin
        start, finish, ws = start_finish_ws[0]
        # Give this job to that worker
        ws.insert(job, start, finish)

    return worker_schedules
