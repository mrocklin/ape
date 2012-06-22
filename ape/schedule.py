from env_manip import pack_many

def var_id(var):
    return var.name

def useful_dicts(sched):

    jobs_of = {machine : [job for job, (time, id) in sched if machine == id]
                              for job, (time, machine) in sched}

    job_runs_on   = {job : machine for job, (time, machine) in sched}

    var_stored_on = {var_id(var) : machine for job, (time, machine) in sched
                                           for var in job.outputs}
    vars = [var for job, _ in sched for var in (job.inputs+job.outputs)]
    var_needed_on = {var_id(var) :
                        [machine for job, (time, machine) in sched
                        if var in job.inputs and job_runs_on[job] == machine]
                     for var in vars}

    return jobs_of, job_runs_on, var_stored_on, var_needed_on

def is_input(var):
    return var.owner is None
def is_output(var):
    c = var.clients
    return len(c) == 1 and c[0][0] == 'output'

def machine_dict_to_code(d):
    """ Converts a machine indexed dict into conditional code

    Turns a dict
    {machine1: [line, line, line], machine2: [line]}

    into a string of a sequence of if statements

    if host == 'machine1':
        line
        line
        line
    if host == 'machine2':
        line
    """
    return ('\n'.join(["if host == '%s':\n"%machine +
        ('\n'.join(["    "+line for line in d[machine]])
            if d[machine] else "    pass")
        for machine in d]))

def gen_code(sched, env_filename, var_shapes):
    """
    Generates all of the code specific to this schedule instance

    Returns a dictionary with code for compilation, instantiation, computation,
    etc... of the schedule.

    This dictionary can be put into the template.py file i.e.

    >>> d = gen_code(sched, "env.dat", {'x': (5,5), 'y':(5,5)})
    >>> generic_code = open('template.py').read()
    >>> code = generic_code%d
    >>> open('result.py', 'w').write(code)

    """
    env_file = open(env_filename, 'w')

    jobs_of, job_runs_on, var_stored_on, var_needed_on = useful_dicts(sched)
    job_ids = {job : i for i, (job, (_, _)) in enumerate(sched)}
    def job_id(job):     return job_ids[job]

    envs = [job for job, (_, _) in sched]
    machines = {machine for _, (_, machine) in sched}
    variable_names = [var.name  for env in envs
                                for var in env.inputs+env.outputs]
    variable_tags = {name : i for i, name in enumerate(variable_names)}
    fn_names = {env: "fn_%d"%i for i, env in enumerate(envs)}
    var_dtype = {var_id(var) : var.dtype for env in envs
                                          for var in env.inputs+env.outputs}
    variable_ids = var_dtype.keys()

    env_file = open(env_filename, 'w')
    pack_many(envs, env_file)
    env_file.close()

    # Creates code like
    # link = mode.linker.accept(envs[4])
    # fn_4 = link.make_function()
    compile_code = {machine : sum([["link = mode.linker.accept(envs[%d])"%i,
                                   "fn_%d = link.make_function()"%i]
                                   for i in map(job_id, jobs_of[machine])], [])
                                   for machine in machines}

    # Creates code like
    # var1 = np.empty((5, 5), dtype="float64")
    var_init_code = {machine :
           ["%s = np.empty(%s, dtype='%s')"%(
                                        vid, var_shapes[vid], var_dtype[vid])
                for vid in variable_ids
                if machine in var_needed_on[vid]]
            for machine in machines}

    # Creates code like
    # recv(var1, 13, "ankaa")
    recv_code = {machine:
            ["recv(%s, %s, '%s')"%(vid, variable_tags[vid], var_stored_on[vid])
                for job in jobs_of[machine]
                for vid, var in zip(map(var_id, job.inputs), job.inputs)
                if not is_input(var)
                and var_stored_on[vid] != machine]
            for machine in machines}

    # Creates code like
    # wait(13)
    # var1, var2 = fn_15(var5, var3)
    # send(var1, 15, "mimosa")
    # send(var2, 18, "arroyitos")
    compute_code = {machine:
        sum([["wait(%s)"%(variable_tags[vid])
                for vid, var in zip(map(var_id, job.inputs), job.inputs)
                if not is_input(var)
                and var_stored_on[vid] != machine]                  +

            [', '.join(map(var_id, job.outputs)) +
             " = %s("%fn_names[job] +
             ', '.join(map(var_id, job.inputs)) + ')']              +

            ["send(%s, %s, '%s')"%(vid, variable_tags[vid], to_machine)
                for vid, var in zip(map(var_id, job.outputs), job.outputs)
                if not is_output(var)
                for to_machine in [m for m in var_needed_on[vid] if m!=machine]]

            for job in jobs_of[machine]], [])
        for machine in machines}

    # Actually, the above objects are dicts that contain lines of code.
    # lets turn that into blocks of code branched on machine id
    var_init_string = machine_dict_to_code(var_init_code)
    compile_string  = machine_dict_to_code(compile_code)
    recv_string     = machine_dict_to_code(recv_code)
    compute_string  = machine_dict_to_code(compute_code)

    return {"env_filename"              : env_filename,
            "compile"                   : compile_string,
            "recv"                      : recv_string,
            "compute"                   : compute_string,
            "variable_initialization"   : var_init_string}
