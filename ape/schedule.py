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

def gen_code(sched, env_filename):
    env_file = open(env_filename, 'w')

    jobs_of, job_runs_on, var_stored_on, var_needed_on = useful_dicts(sched)
    job_ids = {job : i for i, (job, (_, _)) in enumerate(sched)}
    def job_id(job):     return job_ids[job]

    envs = [job for job, (_, _) in sched]
    machines = {machine for _, (_, machine) in sched}
    variable_names = [var.name  for env in envs
                                for var in env.inputs+env.outputs]
    variable_tags = {name : i for i, name in enumerate(variable_names)}
    variable_tag_string = "tag_of = %s"%str(variable_tags)
    fn_names = {env: "fn_%d"%i for i, env in enumerate(envs)}

    def stringify(env):
        return '"""' + pack(env).replace('\n', r'\n') + '"""'

    env_file = open(env_filename, 'w')
    pack_many(envs, env_file)
    env_file.close()

    compile_string =  "\n".join(["link = mode.linker.accept(envs[%d])\n"%i+
                            "fn_%d = link.make_function()"%i
                            for i in range(len(envs))])

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
                    lines = ["recv(%s, %s)"%(vid, source)] + lines
                    lines.append("wait(%s, %s)"%(vid, source))
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
                    lines.append("send(%s, %s)"%(vid, var_needed_on[vid]))

        code[machine] = lines

    code_string = '\n'.join(["\nif host == '%s':\n"%machine +
        '\n'.join(["    "+line for line in code[machine]]) for machine in code])

    return {"env_filename"      : env_filename,
            "compile"           : compile_string,
            "variable_tags"     : variable_tag_string,
            "host_code"         : code_string,
            "host_code_dict"    : code}
