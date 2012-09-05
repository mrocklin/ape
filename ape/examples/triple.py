from ape.util import merge

machine_groups = (('ankaa.cs.uchicago.edu','mimosa.cs.uchicago.edu'),
                  ('baconost.cs.uchicago.edu',),
                  ('baconost.cs.uchicago.edu-gpu',))

cpu_machines = {a:{'type':'cpu'} for group in machine_groups for a in group
                                 if 'gpu' not in a}
gpu_machines = {a:{'type':'gpu', 'host':a.replace('-gpu', '')}
                                 for group in machine_groups for a in group
                                 if 'gpu' in a}
machines = merge(cpu_machines, gpu_machines)

mpi_network = {(a,b):{'type':'mpi'} for a in machines for b in machines if a!=b
                                    if 'gpu' not in a and 'gpu' not in b}
gpu_network = {('baconost.cs.uchicago.edu', 'baconost.cs.uchicago.edu-gpu'):
                    {'type':'togpu'},
               ('baconost.cs.uchicago.edu-gpu', 'baconost.cs.uchicago.edu'):
                    {'type':'fromgpu'}}
network = merge(mpi_network, gpu_network)
