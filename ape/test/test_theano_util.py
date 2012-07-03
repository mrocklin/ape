from ape.theano_util import *

def test_bytes_of_dtype():
    assert bytes_of_dtype('float32') == 4
