from ape.util import unique
from ape.master import sanitize, make_apply
import theano

def test_sanitize():
    x = theano.tensor.matrix('x')
    y = x.T
    z = y + 1
    sanitize((x,), (z,))
    assert all(v.name and '.' not in v.name for v in (x,y,z))
    assert unique((x,y,z))
    print x, y, z

def test_make_apply():
    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    op = theano.tensor.elemwise.Sum()
    job = ((x,), op, y)
    apply = make_apply(*job)
    assert isinstance(apply, theano.Apply)
    assert apply.op == op
    assert apply.inputs[0].name == x.name
    assert apply.outputs[0].name == y.name
