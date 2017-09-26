import numpy as np

class Node(object):
    ids = {}

    def as_constant(self, other):
        if not isinstance(other, Node):
            return Constant(name=None, val=other)
        else:
            return other

    def __add__(self, other):
        return Op("Add", np.add, self, self.as_constant(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Op("Subtract", np.subtract, self, self.as_constant(other))

    def __rsub__(self, other):
        return Op("Subtract", np.subtract, self.as_constant(other), self)

    def __div__(self, other):
        return Op("Divide", np.divide, self, self.as_constant(other))

    def __rdiv__(self, other):
        return Op("Divide", np.divide, self.as_constant(other), self)

    def __mul__(self, other):
        return Op("Multiply", np.multiply, self, self.as_constant(other))

    def __rmul__(self, other):
        return self * other

    def __lt__(self, other):
        return Op("Less_Than", np.less, self, self.as_constant(other))

    def __le__(self, other):
        return Op("Less_Equal", np.less_equal, self, self.as_constant(other))

    def __gt__(self, other):
        return Op("Greater", np.greater, self, self.as_constant(other))

    def __ge__(self, other):
        return Op("Greater_Equal", np.greater_equal, self, self.as_constant(other))

    def __eq__(self, other):
        return Op("equal", np.equal, self, self.as_constant(other))

    def __ne__(self, other):
        return Op("Not_Equal", np.not_equal, self, self.as_constant(other))

    def __hash__(self):
        return id(self)

class Variable(Node):
    def __init__(self, name):
        super(Variable, self).__init__()
        self.name = name

    def eval(self, val_map=None):
        if val_map:
            if self in val_map:
                if isinstance(val_map[self], Constant): return val_map[self]
                else: return Constant(name=None, val=val_map[self])
            else:
                return self
        else:
            return self

    def is_constant(self):
        return False

    def get_val(self):
        raise TypeError("Don't call get_val when Variables don't have a value")

    def __str__(self):
        return "<Variable - Name: {}>".format(self.name)

    __repr__ = __str__


class Constant(Node):
    def __init__(self, val=0.0, name=None):
        super(Constant, self).__init__()
        self.name = name
        self.val = val

    def eval(self, val_map=None):
        if val_map:
            if self in val_map:
                if isinstance(val_map[self], Constant): return val_map[self]
                else: return Constant(name=None, val=val_map[self])
            else:
                return self
        else:
            return self

    def is_constant(self):
        return True

    def get_val(self):
        return self.val

    def __str__(self):
        return "<Constant - Name: {}; Value: {}>".format(self.name, self.val)

    __repr__ = __str__

class Op(Node):
    def __init__(self, name, op, *args, **kwargs):
        super(Op, self).__init__()
        self.name = name
        self.op = op
        self.args = tuple([i if isinstance(i, (Constant, Variable)) else Constant(name=None, val=i) for i in args])
        self.kwargs = {key:i if isinstance(i, (Constant, Variable)) else Constant(name=None, val=i) for key,i in kwargs.items()}

    def eval(self, val_map=None):
        eval_args = [val.eval(val_map) for val in self.args]
        eval_kwargs = {key:val.eval(val_map) for key,val in self.kwargs.items()}
        if all((i.is_constant() for i in eval_args)) and all((i.is_constant() for _,i in eval_kwargs)):
            try:
                return Constant(self.op(*(i.get_val() for i in eval_args), **{key:val.get_val() for key,val in eval_kwargs.items()}))
            except:
                print([i.get_val() for i in eval_args], {key:val.get_val() for key,val in eval_kwargs.items()})
                raise
        else:
            return Op(self.name, self.op, *eval_args, **eval_kwargs)

    def is_constant(self):
        return False

    def get_val(self):
        raise TypeError("Don't call get_val when Variables don't have a value")

    def __str__(self):
        return "<Op - Name: {}; Op: {}; Args: {}; Kwargs: {}>".format(self.name, self.op, ",".join(("{}".format(i) for i in self.args)), ",".join("{}={}".format(key, val) for key,val in self.kwargs))

    __repr__ = __str__

if __name__ == "__main__":
    a = Constant(name="a", val=1)
    b = Variable("b")
    cos = Op("cos", np.cos, a)
    sin = Op("sin", np.sin, b)
    print(cos.eval(), "\n", cos.eval({a:np.pi}))
    print(sin.eval({b:3*np.pi/2}), "\n", sin.eval())
    print(a*b, (a*b).eval({b:2}))
