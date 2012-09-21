
def write_input_file(fgraph, filename, known_shape_strings):
    file = open(filename, 'w')
    file.write('import numpy as np\n')
    for input in fgraph.inputs:
        file.write('%s = np.random.rand(*%s).astype(%s)\n'%(input.name,
            known_shape_strings[input.name], input.dtype))
    file.close()

def write_rankfile(rankdict, filename):
    file = open(filename, 'w')
    for machine, rank in sorted(rankdict.items(), key=lambda (m,r): r):
        file.write("rank %d=%s slot=0\n"%(rank, machine))
    file.close()
