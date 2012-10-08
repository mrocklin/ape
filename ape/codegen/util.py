from ape.env_manip import pack, unpack
import theano

def write_inputs((inputs, outputs), filename, known_shape_strings):
    file = open(filename, 'w')
    file.write('import numpy as np\n')
    for input in inputs:
        file.write("%s = np.random.rand(*%s).astype('%s')\n"%(input.name,
            known_shape_strings[input.name], input.dtype))

    if len(inputs) != 1:
        varstrings = ', '.join(i.name for i in inputs)
    else:
        varstrings = "%s,"%inputs[0]

    file.write('inputs = (%s)'%varstrings)
    file.close()

def read_inputs(filename):
    file = open(filename)
    for line in file:
        exec(line)
    return inputs

def write_rankfile(rankdict, filename):
    file = open(filename, 'w')
    for machine, rank in sorted(rankdict.items(), key=lambda (m,rank): rank):
        file.write("rank %d=%s slot=0\n"%(rank, machine))
    file.close()

def write_hostfile(machines, filename):
    file = open(filename, 'w')
    for machine in machines:
        file.write("%s\n"%machine)
    file.close()

def write_graph((inputs, outputs), filename):
    fgraph = theano.FunctionGraph(*theano.gof.graph.clone(inputs, outputs))
    file = open(filename, 'w')
    pack(fgraph, file)
    file.close()

def read_graph(filename):
    file = open(filename, 'r')
    fgraph = unpack(file)
    file.close()
    return fgraph

def write_sched(sched, filename):
    file = open(filename, 'w')
    for node in sched:
        file.write(str(node)+'\n')
    file.close()

def read_sched(filename):
    file = open(filename, 'r')
    lines = map(lambda s: s.strip(), file.readlines())
    file.close()
    return lines

def sched_to_cmp(sched):
    schedstr = map(str, sched)
    def schedule_cmp(a, b):
        try:
            aind = schedstr.index(str(a))
            bind = schedstr.index(str(b))
            return cmp(aind, bind)
        except ValueError:
            return 0
    return schedule_cmp
