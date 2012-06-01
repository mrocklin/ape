from env_manip import pack_many

def var_id(var):
    return var.name

def useful_dicts(sched):

    jobs_of = {machine : [job for job, (time, id) in sched if machine == id]
                              for job, (time, machine) in sched}

    job_runs_on   = {job : machine for job, (time, machine) in sched}

    var_stored_on = {var_id(var) : machine for job, (time, machine) in sched
                                           for var in job.outputs}

    var_needed_on = {var_id(var) : machine for job, (time, machine) in sched
                                           for var in job.inputs}

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
    return '\n'.join(["if host == '%s':\n"%machine +
        '\n'.join(["    "+line for line in d[machine]]) for machine in d])


def gen_code(sched, env_filename, var_shapes, var_types):
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

    env_file = open(env_filename, 'w')
    pack_many(envs, env_file)
    env_file.close()

    compile_dict = {machine : sum([["link = mode.linker.accept(envs[%d])"%i,
                                   "fn_%d = link.make_function()"%i]
                                   for i in map(job_id, jobs_of[machine])], [])
                                   for machine in machines}
    compile_string = machine_dict_to_code(compile_dict)

    code = dict()
    for machine in machines:
        lines = []
        for job in jobs_of[machine]:
            # Receive variables
            for var in job.inputs:
                if is_input(var):
                    continue
                vid = var_id(var)
                source = var_stored_on[vid]
                if source!=machine:
                    lines = ["recv(%s, %s, '%s')"%(
                        vid, variable_tags[vid], source)] + lines
                    lines.append("wait_on_recv(%s, '%s')"%(
                        variable_tags[vid], source))
            # Compute
            lines.append(', '.join(map(var_id, job.outputs)) +
                         " = %s("%fn_names[job] +
                         ', '.join(map(var_id, job.inputs)) + ')')
            # Send results
            for var in job.outputs:
                if is_output(var):
                    continue
                vid = var_id(var)
                dest = var_needed_on[vid]
                if dest!=machine:
                    lines.append("send(%s, %s, '%s')"%(
                        vid, variable_tags[vid], var_needed_on[vid]))

        code[machine] = lines

    code_string = machine_dict_to_code(code)

    # variable_initialization
    var_init_dict = {machine :
           ["%s = np.empty(%s, dtype=%s)"%(var, var_shapes[var], var_types[var])
                for var in var_needed_on if var_needed_on[var] == machine]
                for machine in machines}

    var_init_string = machine_dict_to_code(var_init_dict)

    return {"env_filename"              : env_filename,
            "compile"                   : compile_string,
            "host_code"                 : code_string,
            "host_code_dict"            : code,
            "variable_initialization"    : var_init_string}
