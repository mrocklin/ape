from ape.env_manip import pack, unpack

def write_inputs(fgraph, filename, known_shape_strings):
    file = open(filename, 'w')
    file.write('import numpy as np\n')
    for input in fgraph.inputs:
        file.write("%s = np.random.rand(*%s).astype('%s')\n"%(input.name,
            known_shape_strings[input.name], input.dtype))

    file.write('inputs = (%s,)\n'%(', '.join(i.name for i in fgraph.inputs)))
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

def write_fgraph(fgraph, filename):
    file = open(filename, 'w')
    pack(fgraph, file)
    file.close()

def read_fgraph(filename):
    file = open(filename, 'r')
    fgraph = unpack(file)
    file.close()
    return fgraph
