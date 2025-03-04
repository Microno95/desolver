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


def fn2(x):
    return D.ar_numpy.sin(x) - 0.5

def fn2_jac(x):
    return D.ar_numpy.cos(x)

fn2.jac = fn2_jac
fn2.root_interval = [0.0, 1.5]


def fn3(x):
    return D.ar_numpy.square(x) - (1 - x)**5

def fn3_jac(x):
    return 2*x + 5*(1 - x)**4

fn3.jac = fn3_jac
fn3.root_interval = [0.0, 1.0]


def fn4(x):
    return D.ar_numpy.exp(-x)*(x - 1) + x

def fn4_jac(x):
    return D.ar_numpy.exp(-x)*(2 - x) + 1

fn4.jac = fn4_jac
fn4.root_interval = [0.0, 1.0]


def fn5(x):
    return 2*x*np.exp(-2) - 2*D.ar_numpy.exp(-2*x) + 1

def fn5_jac(x):
    return 2*np.exp(-2) + 4*D.ar_numpy.exp(-2*x)

fn5.jac = fn5_jac
fn5.root_interval = [0.0, 1.0]

# ---- #

test_fn_param = pytest.mark.parametrize("fn", [fn1, fn2, fn3, fn4, fn5])
