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
        iter(x)
        return True
    except TypeError:
        return False

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
