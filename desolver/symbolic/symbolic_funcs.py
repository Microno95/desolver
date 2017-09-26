from desolver.symbolic import *
import numpy as np
import math

class Sin(Op):
    def __init__(self, x, **kwargs):
        super(Sin, self).__init__("Sin", np.sin, x, **kwargs)

class Cos(Op):
    def __init__(self, x, **kwargs):
        super(Cos, self).__init__("Cos", np.cos, x, **kwargs)

class Tan(Op):
    def __init__(self, x, **kwargs):
        super(Tan, self).__init__("Tan", np.tan, x, **kwargs)

class Arcsin(Op):
    def __init__(self, x, **kwargs):
        super(Arcsin, self).__init__("Arcsin", np.arcsin, x, **kwargs)

class Arccos(Op):
    def __init__(self, x, **kwargs):
        super(Arccos, self).__init__("Arccos", np.arccos, x, **kwargs)

class Arctan(Op):
    def __init__(self, x, **kwargs):
        super(Arctan, self).__init__("Arctan", np.arctan, x, **kwargs)

class Degrees(Op):
    def __init__(self, x, **kwargs):
        super(Degrees, self).__init__("Degrees", np.degrees, x, **kwargs)

class Radians(Op):
    def __init__(self, x, **kwargs):
        super(Radians, self).__init__("Radians", np.radians, x, **kwargs)

class Unwrap(Op):
    def __init__(self, x, **kwargs):
        super(Unwrap, self).__init__("Unwrap", np.unwrap, x, **kwargs)

class Deg2Rad(Op):
    def __init__(self, x, **kwargs):
        super(Deg2Rad, self).__init__("Deg2Rad", np.deg2rad, x, **kwargs)

class Rad2Deg(Op):
    def __init__(self, x, **kwargs):
        super(Rad2Deg, self).__init__("Rad2Deg", np.rad2deg, x, **kwargs)

class Sinh(Op):
    def __init__(self, x, **kwargs):
        super(Sinh, self).__init__("Sinh", np.sinh, x, **kwargs)

class Cosh(Op):
    def __init__(self, x, **kwargs):
        super(Cosh, self).__init__("Cosh", np.cosh, x, **kwargs)

class Tanh(Op):
    def __init__(self, x, **kwargs):
        super(Tanh, self).__init__("Tanh", np.tanh, x, **kwargs)

class Arcsinh(Op):
    def __init__(self, x, **kwargs):
        super(Arcsinh, self).__init__("Arcsinh", np.arcsinh, x, **kwargs)

class Arccosh(Op):
    def __init__(self, x, **kwargs):
        super(Arccosh, self).__init__("Arccosh", np.arccosh, x, **kwargs)

class Arctanh(Op):
    def __init__(self, x, **kwargs):
        super(Arctanh, self).__init__("Arctanh", np.arctanh, x, **kwargs)

class Around(Op):
    def __init__(self, x, **kwargs):
        super(Around, self).__init__("Around", np.around, x, **kwargs)

class Round_(Op):
    def __init__(self, x, **kwargs):
        super(Round_, self).__init__("Round_", np.round_, x, **kwargs)

class Rint(Op):
    def __init__(self, x, **kwargs):
        super(Rint, self).__init__("Rint", np.rint, x, **kwargs)

class Fix(Op):
    def __init__(self, x, **kwargs):
        super(Fix, self).__init__("Fix", np.fix, x, **kwargs)

class Floor(Op):
    def __init__(self, x, **kwargs):
        super(Floor, self).__init__("Floor", np.floor, x, **kwargs)

class Ceil(Op):
    def __init__(self, x, **kwargs):
        super(Ceil, self).__init__("Ceil", np.ceil, x, **kwargs)

class Trunc(Op):
    def __init__(self, x, **kwargs):
        super(Trunc, self).__init__("Trunc", np.trunc, x, **kwargs)

class Prod(Op):
    def __init__(self, x, **kwargs):
        super(Prod, self).__init__("Prod", np.prod, x, **kwargs)

class Sum(Op):
    def __init__(self, x, **kwargs):
        super(Sum, self).__init__("Sum", np.sum, x, **kwargs)

class Nanprod(Op):
    def __init__(self, x, **kwargs):
        super(Nanprod, self).__init__("Nanprod", np.nanprod, x, **kwargs)

class Nansum(Op):
    def __init__(self, x, **kwargs):
        super(Nansum, self).__init__("Nansum", np.nansum, x, **kwargs)

class Cumprod(Op):
    def __init__(self, x, **kwargs):
        super(Cumprod, self).__init__("Cumprod", np.cumprod, x, **kwargs)

class Cumprod(Op):
    def __init__(self, x, **kwargs):
        super(Cumprod, self).__init__("Cumprod", np.cumprod, x, **kwargs)

class Diff(Op):
    def __init__(self, x, **kwargs):
        super(Diff, self).__init__("Diff", np.diff, x, **kwargs)

class Ediff1D(Op):
    def __init__(self, x, **kwargs):
        super(Ediff1D, self).__init__("Ediff1D", np.ediff1d, x, **kwargs)

class Gradient(Op):
    def __init__(self, x, **kwargs):
        super(Gradient, self).__init__("Gradient", np.gradient, x, **kwargs)

class Exp(Op):
    def __init__(self, x, **kwargs):
        super(Exp, self).__init__("Exp", np.exp, x, **kwargs)

class Expm1(Op):
    def __init__(self, x, **kwargs):
        super(Expm1, self).__init__("Expm1", np.expm1, x, **kwargs)

class Exp2(Op):
    def __init__(self, x, **kwargs):
        super(Exp2, self).__init__("Exp2", np.exp2, x, **kwargs)

class Log(Op):
    def __init__(self, x, **kwargs):
        super(Log, self).__init__("Log", np.log, x, **kwargs)

class Log10(Op):
    def __init__(self, x, **kwargs):
        super(Log10, self).__init__("Log10", np.log10, x, **kwargs)

class Log2(Op):
    def __init__(self, x, **kwargs):
        super(Log2, self).__init__("Log2", np.log2, x, **kwargs)

class Log1P(Op):
    def __init__(self, x, **kwargs):
        super(Log1P, self).__init__("Log1P", np.log1p, x, **kwargs)

class I0(Op):
    def __init__(self, x, **kwargs):
        super(I0, self).__init__("I0", np.i0, x, **kwargs)

class Sinc(Op):
    def __init__(self, x, **kwargs):
        super(Sinc, self).__init__("Sinc", np.sinc, x, **kwargs)

class Signbit(Op):
    def __init__(self, x, **kwargs):
        super(Signbit, self).__init__("Signbit", np.signbit, x, **kwargs)

class Spacing(Op):
    def __init__(self, x, **kwargs):
        super(Spacing, self).__init__("Spacing", np.spacing, x, **kwargs)

class Angle(Op):
    def __init__(self, x, **kwargs):
        super(Angle, self).__init__("Angle", np.angle, x, **kwargs)

class Real(Op):
    def __init__(self, x, **kwargs):
        super(Real, self).__init__("Real", np.real, x, **kwargs)

class Imag(Op):
    def __init__(self, x, **kwargs):
        super(Imag, self).__init__("Imag", np.imag, x, **kwargs)

class Conj(Op):
    def __init__(self, x, **kwargs):
        super(Conj, self).__init__("Conj", np.conj, x, **kwargs)

class Sqrt(Op):
    def __init__(self, x, **kwargs):
        super(Sqrt, self).__init__("Sqrt", np.sqrt, x, **kwargs)

class Cbrt(Op):
    def __init__(self, x, **kwargs):
        super(Cbrt, self).__init__("Cbrt", np.cbrt, x, **kwargs)

class Square(Op):
    def __init__(self, x, **kwargs):
        super(Square, self).__init__("Square", np.square, x, **kwargs)

class Absolute(Op):
    def __init__(self, x, **kwargs):
        super(Absolute, self).__init__("Absolute", np.absolute, x, **kwargs)

class Fabs(Op):
    def __init__(self, x, **kwargs):
        super(Fabs, self).__init__("Fabs", np.fabs, x, **kwargs)

class Sign(Op):
    def __init__(self, x, **kwargs):
        super(Sign, self).__init__("Sign", np.sign, x, **kwargs)

class Nan_To_Num(Op):
    def __init__(self, x, **kwargs):
        super(Nan_To_Num, self).__init__("Nan_To_Num", np.nan_to_num, x, **kwargs)

class Real_If_Close(Op):
    def __init__(self, x, **kwargs):
        super(Real_If_Close, self).__init__("Real_If_Close", np.real_if_close, x, **kwargs)

class Hypot(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Hypot, self).__init__("Hypot", np.hypot, x1, x2, **kwargs)

class Arctan2(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Arctan2, self).__init__("Arctan2", np.arctan2, x1, x2, **kwargs)

class Cross(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Cross, self).__init__("Cross", np.cross, x1, x2, **kwargs)

class Logaddexp(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Logaddexp, self).__init__("Logaddexp", np.logaddexp, x1, x2, **kwargs)

class Logaddexp2(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Logaddexp2, self).__init__("Logaddexp2", np.logaddexp2, x1, x2, **kwargs)

class Add(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Add, self).__init__("Add", np.add, x1, x2, **kwargs)

class Multiply(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Multiply, self).__init__("Multiply", np.multiply, x1, x2, **kwargs)

class Divide(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Divide, self).__init__("Divide", np.divide, x1, x2, **kwargs)

class Power(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Power, self).__init__("Power", np.power, x1, x2, **kwargs)

class Subtract(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Subtract, self).__init__("Subtract", np.subtract, x1, x2, **kwargs)

class True_Divide(Op):
    def __init__(self, x1, x2, **kwargs):
        super(True_Divide, self).__init__("True_Divide", np.true_divide, x1, x2, **kwargs)

class Floor_Divide(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Floor_Divide, self).__init__("Floor_Divide", np.floor_divide, x1, x2, **kwargs)

class Fmod(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Fmod, self).__init__("Fmod", np.fmod, x1, x2, **kwargs)

class Mod(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Mod, self).__init__("Mod", np.mod, x1, x2, **kwargs)

class Remainder(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Remainder, self).__init__("Remainder", np.remainder, x1, x2, **kwargs)

class Maximum(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Maximum, self).__init__("Maximum", np.maximum, x1, x2, **kwargs)

class Minimum(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Minimum, self).__init__("Minimum", np.minimum, x1, x2, **kwargs)

class Fmax(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Fmax, self).__init__("Fmax", np.fmax, x1, x2, **kwargs)

class Fmin(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Fmin, self).__init__("Fmin", np.fmin, x1, x2, **kwargs)

class Convolve(Op):
    def __init__(self, x1, x2, **kwargs):
        super(Convolve, self).__init__("Convolve", np.convolve, x1, x2, **kwargs)

class Clip(Op):
    def __init__(self, *args, **kwargs):
        super(Clip, self).__init__("Clip", np.clip, *args, **kwargs)

class Clip(Op):
    def __init__(self, *args, **kwargs):
        super(Clip, self).__init__("Clip", np.clip, *args, **kwargs)

class Interp(Op):
    def __init__(self, *args, **kwargs):
        super(Interp, self).__init__("Interp", np.interp, *args, **kwargs)

class Interp(Op):
    def __init__(self, *args, **kwargs):
        super(Interp, self).__init__("Interp", np.interp, *args, **kwargs)





if __name__ == "__main__":
    print("Numpy version: {}".format(np.__version__))
    x2 = Constant(name=None, val=np.array([1.0, 1.0]))
    x1 = Constant(name=None, val=np.array([1.0, 1.0]))
    x3 = Constant(name=None, val=np.array([1.0, 1.0]))


    try:
        assert (Sin(x1).eval().get_val() == np.sin(x1.get_val())).all()
    except AssertionError:
        print(Sin(x1).eval().get_val(), np.sin(x1.get_val()), "sin\n")
        raise

    try:
        assert (Cos(x1).eval().get_val() == np.cos(x1.get_val())).all()
    except AssertionError:
        print(Cos(x1).eval().get_val(), np.cos(x1.get_val()), "cos\n")
        raise

    try:
        assert (Tan(x1).eval().get_val() == np.tan(x1.get_val())).all()
    except AssertionError:
        print(Tan(x1).eval().get_val(), np.tan(x1.get_val()), "tan\n")
        raise

    try:
        assert (Arcsin(x1).eval().get_val() == np.arcsin(x1.get_val())).all()
    except AssertionError:
        print(Arcsin(x1).eval().get_val(), np.arcsin(x1.get_val()), "arcsin\n")
        raise

    try:
        assert (Arccos(x1).eval().get_val() == np.arccos(x1.get_val())).all()
    except AssertionError:
        print(Arccos(x1).eval().get_val(), np.arccos(x1.get_val()), "arccos\n")
        raise

    try:
        assert (Arctan(x1).eval().get_val() == np.arctan(x1.get_val())).all()
    except AssertionError:
        print(Arctan(x1).eval().get_val(), np.arctan(x1.get_val()), "arctan\n")
        raise

    try:
        assert (Degrees(x1).eval().get_val() == np.degrees(x1.get_val())).all()
    except AssertionError:
        print(Degrees(x1).eval().get_val(), np.degrees(x1.get_val()), "degrees\n")
        raise

    try:
        assert (Radians(x1).eval().get_val() == np.radians(x1.get_val())).all()
    except AssertionError:
        print(Radians(x1).eval().get_val(), np.radians(x1.get_val()), "radians\n")
        raise

    try:
        assert (Unwrap(x1).eval().get_val() == np.unwrap(x1.get_val())).all()
    except AssertionError:
        print(Unwrap(x1).eval().get_val(), np.unwrap(x1.get_val()), "unwrap\n")
        raise

    try:
        assert (Deg2Rad(x1).eval().get_val() == np.deg2rad(x1.get_val())).all()
    except AssertionError:
        print(Deg2Rad(x1).eval().get_val(), np.deg2rad(x1.get_val()), "deg2rad\n")
        raise

    try:
        assert (Rad2Deg(x1).eval().get_val() == np.rad2deg(x1.get_val())).all()
    except AssertionError:
        print(Rad2Deg(x1).eval().get_val(), np.rad2deg(x1.get_val()), "rad2deg\n")
        raise

    try:
        assert (Sinh(x1).eval().get_val() == np.sinh(x1.get_val())).all()
    except AssertionError:
        print(Sinh(x1).eval().get_val(), np.sinh(x1.get_val()), "sinh\n")
        raise

    try:
        assert (Cosh(x1).eval().get_val() == np.cosh(x1.get_val())).all()
    except AssertionError:
        print(Cosh(x1).eval().get_val(), np.cosh(x1.get_val()), "cosh\n")
        raise

    try:
        assert (Tanh(x1).eval().get_val() == np.tanh(x1.get_val())).all()
    except AssertionError:
        print(Tanh(x1).eval().get_val(), np.tanh(x1.get_val()), "tanh\n")
        raise

    try:
        assert (Arcsinh(x1).eval().get_val() == np.arcsinh(x1.get_val())).all()
    except AssertionError:
        print(Arcsinh(x1).eval().get_val(), np.arcsinh(x1.get_val()), "arcsinh\n")
        raise

    try:
        assert (Arccosh(x1).eval().get_val() == np.arccosh(x1.get_val())).all()
    except AssertionError:
        print(Arccosh(x1).eval().get_val(), np.arccosh(x1.get_val()), "arccosh\n")
        raise

    try:
        assert (Arctanh(x1).eval().get_val() == np.arctanh(x1.get_val())).all()
    except AssertionError:
        print(Arctanh(x1).eval().get_val(), np.arctanh(x1.get_val()), "arctanh\n")
        raise

    try:
        assert (Around(x1).eval().get_val() == np.around(x1.get_val())).all()
    except AssertionError:
        print(Around(x1).eval().get_val(), np.around(x1.get_val()), "around\n")
        raise

    try:
        assert (Round_(x1).eval().get_val() == np.round_(x1.get_val())).all()
    except AssertionError:
        print(Round_(x1).eval().get_val(), np.round_(x1.get_val()), "round_\n")
        raise

    try:
        assert (Rint(x1).eval().get_val() == np.rint(x1.get_val())).all()
    except AssertionError:
        print(Rint(x1).eval().get_val(), np.rint(x1.get_val()), "rint\n")
        raise

    try:
        assert (Fix(x1).eval().get_val() == np.fix(x1.get_val())).all()
    except AssertionError:
        print(Fix(x1).eval().get_val(), np.fix(x1.get_val()), "fix\n")
        raise

    try:
        assert (Floor(x1).eval().get_val() == np.floor(x1.get_val())).all()
    except AssertionError:
        print(Floor(x1).eval().get_val(), np.floor(x1.get_val()), "floor\n")
        raise

    try:
        assert (Ceil(x1).eval().get_val() == np.ceil(x1.get_val())).all()
    except AssertionError:
        print(Ceil(x1).eval().get_val(), np.ceil(x1.get_val()), "ceil\n")
        raise

    try:
        assert (Trunc(x1).eval().get_val() == np.trunc(x1.get_val())).all()
    except AssertionError:
        print(Trunc(x1).eval().get_val(), np.trunc(x1.get_val()), "trunc\n")
        raise

    try:
        assert (Prod(x1).eval().get_val() == np.prod(x1.get_val())).all()
    except AssertionError:
        print(Prod(x1).eval().get_val(), np.prod(x1.get_val()), "prod\n")
        raise

    try:
        assert (Sum(x1).eval().get_val() == np.sum(x1.get_val())).all()
    except AssertionError:
        print(Sum(x1).eval().get_val(), np.sum(x1.get_val()), "sum\n")
        raise

    try:
        assert (Nanprod(x1).eval().get_val() == np.nanprod(x1.get_val())).all()
    except AssertionError:
        print(Nanprod(x1).eval().get_val(), np.nanprod(x1.get_val()), "nanprod\n")
        raise

    try:
        assert (Nansum(x1).eval().get_val() == np.nansum(x1.get_val())).all()
    except AssertionError:
        print(Nansum(x1).eval().get_val(), np.nansum(x1.get_val()), "nansum\n")
        raise

    try:
        assert (Cumprod(x1).eval().get_val() == np.cumprod(x1.get_val())).all()
    except AssertionError:
        print(Cumprod(x1).eval().get_val(), np.cumprod(x1.get_val()), "cumprod\n")
        raise

    try:
        assert (Cumprod(x1).eval().get_val() == np.cumprod(x1.get_val())).all()
    except AssertionError:
        print(Cumprod(x1).eval().get_val(), np.cumprod(x1.get_val()), "cumprod\n")
        raise

    try:
        assert (Diff(x1).eval().get_val() == np.diff(x1.get_val())).all()
    except AssertionError:
        print(Diff(x1).eval().get_val(), np.diff(x1.get_val()), "diff\n")
        raise

    try:
        assert (Ediff1D(x1).eval().get_val() == np.ediff1d(x1.get_val())).all()
    except AssertionError:
        print(Ediff1D(x1).eval().get_val(), np.ediff1d(x1.get_val()), "ediff1d\n")
        raise

    try:
        assert (Gradient(x1).eval().get_val() == np.gradient(x1.get_val())).all()
    except AssertionError:
        print(Gradient(x1).eval().get_val(), np.gradient(x1.get_val()), "gradient\n")
        raise

    try:
        assert (Exp(x1).eval().get_val() == np.exp(x1.get_val())).all()
    except AssertionError:
        print(Exp(x1).eval().get_val(), np.exp(x1.get_val()), "exp\n")
        raise

    try:
        assert (Expm1(x1).eval().get_val() == np.expm1(x1.get_val())).all()
    except AssertionError:
        print(Expm1(x1).eval().get_val(), np.expm1(x1.get_val()), "expm1\n")
        raise

    try:
        assert (Exp2(x1).eval().get_val() == np.exp2(x1.get_val())).all()
    except AssertionError:
        print(Exp2(x1).eval().get_val(), np.exp2(x1.get_val()), "exp2\n")
        raise

    try:
        assert (Log(x1).eval().get_val() == np.log(x1.get_val())).all()
    except AssertionError:
        print(Log(x1).eval().get_val(), np.log(x1.get_val()), "log\n")
        raise

    try:
        assert (Log10(x1).eval().get_val() == np.log10(x1.get_val())).all()
    except AssertionError:
        print(Log10(x1).eval().get_val(), np.log10(x1.get_val()), "log10\n")
        raise

    try:
        assert (Log2(x1).eval().get_val() == np.log2(x1.get_val())).all()
    except AssertionError:
        print(Log2(x1).eval().get_val(), np.log2(x1.get_val()), "log2\n")
        raise

    try:
        assert (Log1P(x1).eval().get_val() == np.log1p(x1.get_val())).all()
    except AssertionError:
        print(Log1P(x1).eval().get_val(), np.log1p(x1.get_val()), "log1p\n")
        raise

    try:
        assert (I0(x1).eval().get_val() == np.i0(x1.get_val())).all()
    except AssertionError:
        print(I0(x1).eval().get_val(), np.i0(x1.get_val()), "i0\n")
        raise

    try:
        assert (Sinc(x1).eval().get_val() == np.sinc(x1.get_val())).all()
    except AssertionError:
        print(Sinc(x1).eval().get_val(), np.sinc(x1.get_val()), "sinc\n")
        raise

    try:
        assert (Signbit(x1).eval().get_val() == np.signbit(x1.get_val())).all()
    except AssertionError:
        print(Signbit(x1).eval().get_val(), np.signbit(x1.get_val()), "signbit\n")
        raise

    try:
        assert (Spacing(x1).eval().get_val() == np.spacing(x1.get_val())).all()
    except AssertionError:
        print(Spacing(x1).eval().get_val(), np.spacing(x1.get_val()), "spacing\n")
        raise

    try:
        assert (Angle(x1).eval().get_val() == np.angle(x1.get_val())).all()
    except AssertionError:
        print(Angle(x1).eval().get_val(), np.angle(x1.get_val()), "angle\n")
        raise

    try:
        assert (Real(x1).eval().get_val() == np.real(x1.get_val())).all()
    except AssertionError:
        print(Real(x1).eval().get_val(), np.real(x1.get_val()), "real\n")
        raise

    try:
        assert (Imag(x1).eval().get_val() == np.imag(x1.get_val())).all()
    except AssertionError:
        print(Imag(x1).eval().get_val(), np.imag(x1.get_val()), "imag\n")
        raise

    try:
        assert (Conj(x1).eval().get_val() == np.conj(x1.get_val())).all()
    except AssertionError:
        print(Conj(x1).eval().get_val(), np.conj(x1.get_val()), "conj\n")
        raise

    try:
        assert (Sqrt(x1).eval().get_val() == np.sqrt(x1.get_val())).all()
    except AssertionError:
        print(Sqrt(x1).eval().get_val(), np.sqrt(x1.get_val()), "sqrt\n")
        raise

    try:
        assert (Cbrt(x1).eval().get_val() == np.cbrt(x1.get_val())).all()
    except AssertionError:
        print(Cbrt(x1).eval().get_val(), np.cbrt(x1.get_val()), "cbrt\n")
        raise

    try:
        assert (Square(x1).eval().get_val() == np.square(x1.get_val())).all()
    except AssertionError:
        print(Square(x1).eval().get_val(), np.square(x1.get_val()), "square\n")
        raise

    try:
        assert (Absolute(x1).eval().get_val() == np.absolute(x1.get_val())).all()
    except AssertionError:
        print(Absolute(x1).eval().get_val(), np.absolute(x1.get_val()), "absolute\n")
        raise

    try:
        assert (Fabs(x1).eval().get_val() == np.fabs(x1.get_val())).all()
    except AssertionError:
        print(Fabs(x1).eval().get_val(), np.fabs(x1.get_val()), "fabs\n")
        raise

    try:
        assert (Sign(x1).eval().get_val() == np.sign(x1.get_val())).all()
    except AssertionError:
        print(Sign(x1).eval().get_val(), np.sign(x1.get_val()), "sign\n")
        raise

    try:
        assert (Nan_To_Num(x1).eval().get_val() == np.nan_to_num(x1.get_val())).all()
    except AssertionError:
        print(Nan_To_Num(x1).eval().get_val(), np.nan_to_num(x1.get_val()), "nan_to_num\n")
        raise

    try:
        assert (Real_If_Close(x1).eval().get_val() == np.real_if_close(x1.get_val())).all()
    except AssertionError:
        print(Real_If_Close(x1).eval().get_val(), np.real_if_close(x1.get_val()), "real_if_close\n")
        raise

    try:
        assert (Hypot(x1, x2).eval().get_val() == np.hypot(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Hypot(x1, x2).eval().get_val(), np.hypot(x1.get_val(), x2.get_val()), "hypot\n")
        raise

    try:
        assert (Arctan2(x1, x2).eval().get_val() == np.arctan2(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Arctan2(x1, x2).eval().get_val(), np.arctan2(x1.get_val(), x2.get_val()), "arctan2\n")
        raise

    try:
        assert (Cross(x1, x2).eval().get_val() == np.cross(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Cross(x1, x2).eval().get_val(), np.cross(x1.get_val(), x2.get_val()), "cross\n")
        raise

    try:
        assert (Logaddexp(x1, x2).eval().get_val() == np.logaddexp(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Logaddexp(x1, x2).eval().get_val(), np.logaddexp(x1.get_val(), x2.get_val()), "logaddexp\n")
        raise

    try:
        assert (Logaddexp2(x1, x2).eval().get_val() == np.logaddexp2(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Logaddexp2(x1, x2).eval().get_val(), np.logaddexp2(x1.get_val(), x2.get_val()), "logaddexp2\n")
        raise

    try:
        assert (Add(x1, x2).eval().get_val() == np.add(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Add(x1, x2).eval().get_val(), np.add(x1.get_val(), x2.get_val()), "add\n")
        raise

    try:
        assert (Multiply(x1, x2).eval().get_val() == np.multiply(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Multiply(x1, x2).eval().get_val(), np.multiply(x1.get_val(), x2.get_val()), "multiply\n")
        raise

    try:
        assert (Divide(x1, x2).eval().get_val() == np.divide(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Divide(x1, x2).eval().get_val(), np.divide(x1.get_val(), x2.get_val()), "divide\n")
        raise

    try:
        assert (Power(x1, x2).eval().get_val() == np.power(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Power(x1, x2).eval().get_val(), np.power(x1.get_val(), x2.get_val()), "power\n")
        raise

    try:
        assert (Subtract(x1, x2).eval().get_val() == np.subtract(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Subtract(x1, x2).eval().get_val(), np.subtract(x1.get_val(), x2.get_val()), "subtract\n")
        raise

    try:
        assert (True_Divide(x1, x2).eval().get_val() == np.true_divide(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(True_Divide(x1, x2).eval().get_val(), np.true_divide(x1.get_val(), x2.get_val()), "true_divide\n")
        raise

    try:
        assert (Floor_Divide(x1, x2).eval().get_val() == np.floor_divide(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Floor_Divide(x1, x2).eval().get_val(), np.floor_divide(x1.get_val(), x2.get_val()), "floor_divide\n")
        raise

    try:
        assert (Fmod(x1, x2).eval().get_val() == np.fmod(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Fmod(x1, x2).eval().get_val(), np.fmod(x1.get_val(), x2.get_val()), "fmod\n")
        raise

    try:
        assert (Mod(x1, x2).eval().get_val() == np.mod(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Mod(x1, x2).eval().get_val(), np.mod(x1.get_val(), x2.get_val()), "mod\n")
        raise

    try:
        assert (Remainder(x1, x2).eval().get_val() == np.remainder(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Remainder(x1, x2).eval().get_val(), np.remainder(x1.get_val(), x2.get_val()), "remainder\n")
        raise

    try:
        assert (Maximum(x1, x2).eval().get_val() == np.maximum(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Maximum(x1, x2).eval().get_val(), np.maximum(x1.get_val(), x2.get_val()), "maximum\n")
        raise

    try:
        assert (Minimum(x1, x2).eval().get_val() == np.minimum(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Minimum(x1, x2).eval().get_val(), np.minimum(x1.get_val(), x2.get_val()), "minimum\n")
        raise

    try:
        assert (Fmax(x1, x2).eval().get_val() == np.fmax(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Fmax(x1, x2).eval().get_val(), np.fmax(x1.get_val(), x2.get_val()), "fmax\n")
        raise

    try:
        assert (Fmin(x1, x2).eval().get_val() == np.fmin(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Fmin(x1, x2).eval().get_val(), np.fmin(x1.get_val(), x2.get_val()), "fmin\n")
        raise

    try:
        assert (Convolve(x1, x2).eval().get_val() == np.convolve(x1.get_val(), x2.get_val())).all()
    except AssertionError:
        print(Convolve(x1, x2).eval().get_val(), np.convolve(x1.get_val(), x2.get_val()), "convolve\n")
        raise

    try:
        assert (Clip(x1, x2, x3).eval().get_val() == np.clip(x1.get_val(), x2.get_val(), x3.get_val())).all()
    except AssertionError:
        print(Clip(x1, x2, x3).eval().get_val(), np.clip(x1.get_val(), x2.get_val(), x3.get_val()), "clip\n")
        raise

    try:
        assert (Interp(x1, x2, x3).eval().get_val() == np.interp(x1.get_val(), x2.get_val(), x3.get_val())).all()
    except AssertionError:
        print(Interp(x1, x2, x3).eval().get_val(), np.interp(x1.get_val(), x2.get_val(), x3.get_val()), "interp\n")
        raise
