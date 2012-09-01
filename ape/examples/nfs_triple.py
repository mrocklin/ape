machine_groups = (('ankaa.cs.uchicago.edu','mimosa.cs.uchicago.edu'),
                  ('milkweed.cs.uchicago.edu',))
machines = {a for group in machine_groups for a in group}
network = {(a,b):{'type':'mpi'} for a in machines for b in machines if a!=b}
