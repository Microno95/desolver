import desolver.backend as D
import pytest
import numpy as np

# Defines test functions to benchmark against #

# From https://anthonylloyd.github.io/public/root/1995_Algorithm_748_Enclosing_Zeros_of_Continuous_Functions.pdf

def fn1(x):
    return D.ar_numpy.sin(x) - x/2

def fn1_jac(x):
    return D.ar_numpy.cos(x) - 1/2

fn1.jac = fn1_jac
fn1.root_interval = [D.pi/2, D.pi]
fn1.min_precision = 16


def fn2(x):
    return D.ar_numpy.sin(x) - 0.5

def fn2_jac(x):
    return D.ar_numpy.cos(x)

fn2.jac = fn2_jac
fn2.root_interval = [0.0, 1.5]
fn2.min_precision = 16


def fn3(x):
    return D.ar_numpy.square(x) - (1 - x)**5

def fn3_jac(x):
    return 2*x + 5*(1 - x)**4

fn3.jac = fn3_jac
fn3.root_interval = [0.0, 1.0]
fn3.min_precision = 16

def generate_problems_kind_10(n):
    def fn4(x):
        return D.ar_numpy.exp(-n*x)*(x - 1) + x**n

    def fn4_jac(x):
        return -n*D.ar_numpy.exp(-n*x)*(x - 1) + D.ar_numpy.exp(-n*x) + n*x**(n-1)

    fn4.jac = fn4_jac
    fn4.root_interval = [0.0, 1.0]
    fn4.min_precision = 16

    return fn4

def fn5(x):
    return 2*x*np.exp(-2) - 2*D.ar_numpy.exp(-2*x) + 1

def fn5_jac(x):
    return 2*np.exp(-2) + 4*D.ar_numpy.exp(-2*x)

fn5.jac = fn5_jac
fn5.root_interval = [0.0, 1.0]
fn5.min_precision = 16

def fn6(x):
    return np.where(
        x == 0,
        0.0,
        x*np.exp(-x**-2)
    )

def fn6_jac(x):
    return np.where(
        x == 0,
        0.0,
        np.exp(-1/x**2) + 2*np.exp(-1/x**2)/x**2
    )

fn6.jac = fn6_jac
fn6.root_interval = [-1.0, 4.0]
fn6.min_precision = 16

def generate_problems_kind_11(n):
    def fn7(x):
        return (n*x - 1)/((n-1)*x)

    def fn7_jac(x):
        return 1/(x**2*(n - 1))

    fn7.jac = fn7_jac
    fn7.root_interval = [0.01, 1.0]
    fn7.min_precision = 16

    return fn7

def generate_problems_kind_8(n):
    def fn8(x):
        return x**2 - (1 - x)**n

    def fn8_jac(x):
        return 2*x + n*(1 - x)**(n-1)

    fn8.jac = fn8_jac
    fn8.root_interval = [0.0, 1.0]
    fn8.min_precision = 16

    return fn8

# ---- #

test_fn_param = pytest.mark.parametrize("fn", [fn1, fn2, fn3, *[generate_problems_kind_10(n) for n in [1, 5, 10, 15, 20]], fn5, fn6, 
                                               *[generate_problems_kind_11(n) for n in [2]], *[generate_problems_kind_8(n) for n in [2,5,10,15,20]]])
