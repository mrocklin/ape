class Schedule(object):

    def jobs_in_order(self):
        return sorted(self.start_time_of.items(), key=lambda (j,t):t)

    def code(self):
        all_code = {machine:[] for machine in self.system.machines}
        for job, time in self.jobs_in_order():
            machine = self.runs_on[job]
            # Receive variables from all relevant parent jobs
            for from_job in job.parents:
                for var in set(job.inputs).intersection(set(from_job.outputs)):
                    A, B = self.runs_on[from_job], self.runs_on[job]
                    receive_code = self.system.comm.receive_wait_code(A, B, var)
                    for m, code in receive_code.items():
                        if code not in all_code[m]: # duplicates possible
                            all_code[m].append(code)

            # Run this job
            all_code[machine].append(machine._run_code(job))

            # Send variables to all relevant child jobs
            for to_job in job.children:
                # send each shared variable individually between jobs
                for var in set(job.outputs).intersection(set(to_job.inputs)):
                    A, B = self.runs_on[job], self.runs_on[to_job]
                    transfer_code = self.system.comm.transfer_code(A, B, var)
                    for m, code in transfer_code.items():
                        if code not in all_code[m]: # duplicates possible
                            all_code[m].append(code)

        return all_code

class HEFTSchedule(Schedule):
    def __init__(self, computation, system, sched):
        runs_on = dict()
        jobs_on = dict()
        start_time_of = dict()
        finish_time_of = dict()
        for machine in system.machines:
            jobs_on[machine] = []
        for ws in sched:
            for job, start, finish in ws.task_list:
                runs_on[job] = ws.worker
                jobs_on[ws.worker].append(job)
                start_time_of[job] = start
                finish_time_of[job] = finish

        self.runs_on = runs_on
        self.jobs_on = jobs_on
        self.start_time_of = start_time_of
        self.computation = computation
        self.system = system


class TompkinsSchedule(Schedule):
    def __init__(self, computation, system, sched):
        runs_on = dict()
        jobs_on = dict()
        start_time_of = dict()
        for machine in system.machines:
            jobs_on[machine] = []
        for job, (start_time, machine) in sched:
            runs_on[job] = machine
            start_time_of[job] = start_time
            jobs_on[machine].append(job)

        self.runs_on = runs_on
        self.jobs_on = jobs_on
        self.start_time_of = start_time_of
        self.computation = computation
        self.system = system
