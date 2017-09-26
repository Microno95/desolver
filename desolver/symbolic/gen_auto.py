unary_funcs = [("np", "sin"),("np", "cos"),("np", "tan"),("np", "arcsin"),("np", "arccos"),("np", "arctan"),
               ("np", "degrees"),("np", "radians"),("np", "unwrap"),("np", "deg2rad"),("np", "rad2deg"),
               ("np", "sinh"),("np", "cosh"),("np", "tanh"),("np", "arcsinh"),("np", "arccosh"),("np", "arctanh"),
               ("np", "around"),("np", "round_"),("np", "rint"),("np", "fix"),("np", "floor"),("np", "ceil"),
               ("np", "trunc"),("np", "prod"),("np", "sum"),("np", "nanprod"),("np", "nansum"),("np", "cumprod"),("np", "cumprod"),
               ("np", "diff"),("np", "ediff1d"),("np", "gradient"),
               ("np", "exp"),("np", "expm1"),("np", "exp2"),("np", "log"),("np", "log10"),("np", "log2"),("np", "log1p"),
               ("np", "i0"),("np", "sinc"),("np", "signbit"),("np", "spacing"),("np", "angle"),
               ("np", "real"),("np", "imag"),("np", "conj"),("np", "sqrt"),("np", "cbrt"),("np", "square"),("np", "absolute"),
               ("np", "fabs"),("np", "sign"),("np", "nan_to_num"),("np", "real_if_close")]
binary_funcs = [("np", "hypot"),("np", "arctan2"),("np", "cross"),("np", "logaddexp"),("np", "logaddexp2"),
                ("np", "add"),("np", "multiply"),("np", "divide"),("np", "power"),("np", "subtract"),
                ("np", "true_divide"),("np", "floor_divide"),("np", "fmod"),("np", "mod"),("np", "remainder"),
                ("np", "maximum"),("np", "minimum"),("np", "fmax"),("np", "fmin"),("np", "convolve"),]
ternary_funcs = [("np", "clip"),("np", "interp")]

base_unary_func_str = \
"""class {}(Op):
    def __init__(self, x, **kwargs):
        super({}, self).__init__("{}", {}, x, **kwargs)"""

base_binary_func_str = \
"""class {}(Op):
    def __init__(self, x1, x2, **kwargs):
        super({}, self).__init__("{}", {}, x1, x2, **kwargs)"""

base_ternary_func_str = \
"""class {}(Op):
    def __init__(self, *args, **kwargs):
        super({}, self).__init__("{}", {}, *args, **kwargs)"""

def main():
    main_test_section = """
if __name__ == "__main__":
    print("Numpy version: {}".format(np.__version__))
    x2 = Constant(name=None, val=np.array([1.0, 1.0]))
    x1 = Constant(name=None, val=np.array([1.0, 1.0]))
    x3 = Constant(name=None, val=np.array([1.0, 1.0]))

    """
    with open("symbolic_funcs.py", "w") as fle:
        fle.write("""from symbolic import *\nimport numpy as np\nimport math\n\n""")
        for func in unary_funcs:
            func_string = base_unary_func_str.format(func[1].title(), func[1].title(), func[1].title(), func[0]+"."+func[1])
            func_string = func_string + "\n\n"
            main_test_section = main_test_section + \
"""
    try:
        assert ({}(x1).eval().get_val() == {}(x1.get_val())).all()
    except AssertionError:
        print({}(x1).eval().get_val(), {}(x1.get_val()), "{}\\n")
        raise
""".format(func[1].title(), func[0]+"."+func[1], func[1].title(), func[0]+"."+func[1], func[1])
            fle.write(func_string)


        for func in binary_funcs:
            func_string = base_binary_func_str.format(func[1].title(), func[1].title(), func[1].title(), func[0]+"."+func[1])
            func_string = func_string + "\n\n"
            main_test_section = main_test_section + \
"""
    try:
        assert ({}(x1, x2).eval().get_val() == {}(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print({}(x1, x2).eval().get_val(), {}(x1.get_val(), x2.get_val()), "{}\\n")
        raise
""".format(func[1].title(), func[0]+"."+func[1], func[1].title(), func[0]+"."+func[1], func[1])
            fle.write(func_string)


        for func in ternary_funcs:
            func_string = base_ternary_func_str.format(func[1].title(), func[1].title(), func[1].title(), func[0]+"."+func[1])
            func_string = func_string + "\n\n"
            main_test_section = main_test_section + \
"""
    try:
        assert ({}(x1, x2, x3).eval().get_val() == {}(x1.get_val(), x2.get_val(), x3.get_val())).all()
    except AssertionError:
        print({}(x1, x2, x3).eval().get_val(), {}(x1.get_val(), x2.get_val(), x3.get_val()), "{}\\n")
        raise
""".format(func[1].title(), func[0]+"."+func[1], func[1].title(), func[0]+"."+func[1], func[1])
            fle.write(func_string)
            fle.write(func_string)
        fle.write("\n\n\n" + main_test_section)

if __name__ == "__main__":
    main()
