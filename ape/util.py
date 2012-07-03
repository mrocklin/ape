def chain(*fns):
    """ Composes the functions together so that

    >>> chain(f, g, h)(*args) ==  h(g(f(*args)))
    True

    Inputs follow through the functions from left to right
    """
    def f(*args):
        for sub_fn in fns:
            if not iterable(args): args = [args]
            args = sub_fn(*args)
        return args
    return f

def iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False
