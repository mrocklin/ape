def set_union(sets):
    A = set()
    for s in sets:
        A = A.union(s)
    return A

def host_name():
    import os
    return os.popen('uname -n').read().strip()
