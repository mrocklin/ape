import ast

def chain(*fns):
    """ Composes the functions together so that

    >>> chain(f, g, h)(*args) ==  h(g(f(*args)))
    True

    Inputs follow through the functions from left to right
    """
    def f(*args):
        for sub_fn in fns:
            if not iterable(args): args = [args]
            try:
                if sub_fn.func_code.co_argcount == 1 and len(args)>1:
                    args = sub_fn(args) # function may want an iterable
                else:
                    args = sub_fn(*args) # expand results
            except AttributeError: # some builtins don't have func_code
                try:    args = sub_fn(*args) # just try blindly
                except: args = sub_fn(args)
        return args
    return f

def iterable(x):
    try:
        iter(x).next()
        return True
    except (TypeError):
        return False
    except StopIteration:
        return True

prod = lambda L : reduce(lambda a,b:a*b, L, 1)

def save_dict(filename, d, pretty=True):
    file = open(filename,'w')
    if pretty:
        file.write("{%s}"%(',\n'.join("%s: %s"%(key, value)
                                        for key,value in d.items())))
    else:
        file.write(str(d))
    file.close()

def load_dict(filename):
    file = open(filename); d_string = file.read(); file.close()
    return ast.literal_eval(d_string)

def dearrayify(x):
    from numpy import ndarray
    if isinstance(x, ndarray) and x.shape == ():
        return x.sum()
    if isinstance(x, tuple):
        return tuple(map(dearrayify, x))
    if isinstance(x, dict):
        return {k:dearrayify(val) for k, val in x.items()}
    return x


def merge_values(d, e):
    if not all(isinstance(val, dict) for val in (d.values() + e.values())):
        raise TypeError("Must be dicts of value-type dict")
    if not d.keys() == e.keys():
        raise ValueError("Dicts must have same keys")

    return {key: merge(d[key], e[key]) for key in d}

def merge(*args):
    return dict(sum([arg.items() for arg in args], []))

def intersection(a, b):
    return set(a).intersection(set(b))

def unique(c):
    return len(c) == len(set(c))

def remove(pred, L):
    out = []
    for item in L:
        if not pred(item):
            out.append(item)
    return out
