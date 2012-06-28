import ast
import os

def test_mpi_timing():
    hosts = ['ankaa.cs.uchicago.edu', 'mimosa.cs.uchicago.edu']

    file = open('_machinefile.txt', 'w')
    file.write('\n'.join(hosts))
    file.close()

    ns = [10, 20, 50, 100, 1000]

    s = os.popen('''mpiexec -np 2 -machinefile _machinefile.txt python mpi_timing.py "%s" %s %s'''%(ns, hosts[0], hosts[1]))

    values = ast.literal_eval(s.read())
    assert all(isinstance(n, int) for n,time in values)
    assert all(isinstance(time, float) for n,time in values)
