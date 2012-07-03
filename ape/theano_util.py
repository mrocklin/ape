import theano

known_dtypes = {'float32': 4, 'float64':8, 'int32':4, 'int64':8}
def bytes_of_dtype(dtype):
    if dtype in known_dtypes:
        return known_dtypes[dtype]
    raise NotImplementedError("Don't know how many bytes are in dtype%s"%dtype)
