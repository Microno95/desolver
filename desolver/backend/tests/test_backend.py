import pytest
import os
import numpy as np
import desolver as de
import desolver.backend as D


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_contract_first_ndims_case_1(ffmt):
    arr1 = D.array([[2.0, 1.0], [1.0, 0.0]])
    arr2 = D.array([[1.0, 1.0], [-1.0, 1.0]])

    arr3 = D.contract_first_ndims(arr1, arr2, 1)

    true_arr3 = D.array([1.0, 1.0])

    assert (D.norm(arr3 - true_arr3) <= 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_contract_first_ndims_case_2(ffmt):
    arr1 = D.array([[2.0, 1.0], [1.0, 0.0]])
    arr2 = D.array([[1.0, 1.0], [-1.0, 1.0]])

    arr4 = D.contract_first_ndims(arr1, arr2, 2)

    true_arr4 = D.array(2.)

    assert (D.norm(arr4 - true_arr4) <= 2 * D.epsilon())


def test_contract_first_ndims_reverse_order():
    a = D.array([1.0, 2.0])
    b = D.array([[1.0, 2.0], [2.0, 3.0]])
    D.contract_first_ndims(b, a, n=1)


def test_contract_first_ndims_shape_too_small():
    with pytest.raises(ValueError):
        a = D.array([1.0, 2.0])
        b = D.array([[1.0, 2.0], [2.0, 3.0]])
        D.contract_first_ndims(a, b, n=2)


def test_wrong_float_fmt():
    with pytest.raises(ValueError):
        D.set_float_fmt('potato')

expected_eps = {'float16': 5e-3, 'float32': 5e-7, 'float64': 5e-16, 'gdual_double': 5e-16,
                'gdual_vdouble': 5e-16, 'gdual_real128': 5e-16}


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_default_float_fmt(ffmt):
    D.set_float_fmt(ffmt)
    assert (D.array(1.0).dtype == D.float_fmts[D.float_fmt()])


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_set_float_fmt(ffmt):
    D.set_float_fmt(ffmt)
    assert (D.float_fmt() == str(ffmt))


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_float_fmt_epsilon(ffmt):
    D.set_float_fmt(ffmt)
    assert (D.epsilon() == expected_eps[str(ffmt)])


def test_available_float_fmt_return_type():
    assert (isinstance(D.available_float_fmt(), list))


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_float_fmt_epsilon(ffmt):
    test_array = np.array([1], dtype=np.int64)
    D.set_float_fmt(ffmt)
    assert (D.cast_to_float_fmt(test_array).dtype == str(ffmt))


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_pi_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    assert (np.pi - 2 * D.epsilon() <= D.to_float(D.pi) <= np.pi + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_e_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    assert (np.e - 2 * D.epsilon() <= D.to_float(D.e) <= np.e + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_euler_gamma_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    assert (np.euler_gamma - 2 * D.epsilon() <= D.to_float(D.euler_gamma) <= np.euler_gamma + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_cos_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (-2 * D.epsilon() <= D.cos(pi) + 1 <= 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_sin_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (-2 * D.epsilon() <= D.sin(pi) <= 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_tan_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (-2 * D.epsilon() <= D.tan(pi) <= 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_acos_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    assert (D.abs(D.acos(D.to_float(1))) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_asin_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    ref = D.to_float(D.pi / 2)
    assert (D.abs(D.asin(D.to_float(1)) - ref) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_atan_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    ref = D.to_float(D.pi / 4)
    assert (D.abs(D.atan(D.to_float(1)) - ref) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_atan2_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    ref = D.to_float(D.pi / 4)
    assert (D.abs(D.atan2(D.to_float(1), D.to_float(1)) - ref) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_cosh_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.abs(D.cosh(pi) - np.cosh(pi)) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_sinh_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.abs(D.sinh(pi) - np.sinh(pi)) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_tanh_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.abs(D.tanh(pi) - np.tanh(pi)) <= 2*D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_neg_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (-3.141592653589793 - 2 * D.epsilon() <= D.neg(pi) <= -3.141592653589793 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_pow_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (31.00627668029982 - 10 * D.epsilon() <= D.pow(pi, 3) <= 31.00627668029982 + 10 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_abs_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (3.141592653589793 - 2 * D.epsilon() <= D.abs(pi) <= 3.141592653589793 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_sqrt_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.77245385090551603 - 2 * D.epsilon() <= D.sqrt(pi) <= 1.77245385090551603 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_exp_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (23.1406926327792690 - 10 * D.epsilon() <= D.exp(pi) <= 23.1406926327792690 + 10 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_expm1_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (22.1406926327792690 - 10 * D.epsilon() <= D.expm1(pi) <= 22.1406926327792690 + 10 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_log_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.14472988584940017 - 2 * D.epsilon() <= D.log(pi) <= 1.14472988584940017 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_log10_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.49714987269413385 - 2 * D.epsilon() <= D.log10(pi) <= 0.49714987269413385 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_log1p_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.42108041279429263 - 2 * D.epsilon() <= D.log1p(pi) <= 1.42108041279429263 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_log2_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.65149612947231880 - 2 * D.epsilon() <= D.log2(pi) <= 1.65149612947231880 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_add_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (4.14159265358979324 - 2 * D.epsilon() <= D.add(pi, 1) <= 4.14159265358979324 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_sub_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (2.14159265358979324 - 2 * D.epsilon() <= D.sub(pi, 1) <= 2.14159265358979324 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_div_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.5707963267948966192 - 2 * D.epsilon() <= D.div(pi, 2) <= 1.5707963267948966192 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_mul_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (6.2831853071795864769 - 2 * D.epsilon() <= D.mul(pi, 2) <= 6.2831853071795864769 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_reciprocal_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.31830988618379067 - 2 * D.epsilon() <= D.reciprocal(pi) <= 0.31830988618379067 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_erf_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.9999911238536324 - 2 * D.epsilon() <= D.erf(pi) <= 0.9999911238536324 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_erfc_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (8.8761463676416054e-6 - 2 * D.epsilon() <= D.erfc(pi) <= 8.8761463676416054e-6 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_sigmoid_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.9585761678336372 - 2 * D.epsilon() <= D.sigmoid(pi) <= 0.9585761678336372 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_rsqrt_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.5641895835477563 - 2 * D.epsilon() <= D.rsqrt(pi) <= 0.5641895835477563 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_lerp_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (pi + 0.5 - 2 * D.epsilon() <= D.lerp(pi, pi + 1, 0.5) <= pi + 0.5 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_lerp_within_tolerance_out(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    out = D.copy(pi)
    D.lerp(pi, pi + 1, 0.5, out=out)
    assert (pi + 0.5 - 2 * D.epsilon() <= out <= pi + 0.5 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcdiv_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (pi + (1 * (3 / 2)) - 2 * D.epsilon() <= D.addcdiv(pi, D.to_float(3), D.to_float(2), value=1) <= pi + (1 * (3 / 2)) + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcdiv_error(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    with pytest.raises((ValueError, TypeError)):
        D.addcdiv(pi, value=1)


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcdiv_within_tolerance_out(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    out = D.copy(pi)
    D.addcdiv(pi, D.to_float(3), D.to_float(2), value=1, out=out)
    assert (pi + (1 * (3 / 2)) - 2 * D.epsilon() <= out <= pi + (1 * (3 / 2)) + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcmul_error(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    with pytest.raises(ValueError):
        D.addcmul(pi, value=1)


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcmul_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (pi + (1 * (3 * 2)) - 2 * D.epsilon() <= D.addcmul(pi, D.to_float(3), D.to_float(2), value=1) <= pi + (1 * (3 * 2)) + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcmul_error(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    with pytest.raises((ValueError, TypeError)):
        D.addcmul(pi, value=1)


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_addcmul_within_tolerance_out(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    out = D.copy(pi)
    D.addcmul(pi, D.to_float(3), D.to_float(2), value=1, out=out)
    assert (pi + (1 * (3 * 2)) - 2 * D.epsilon() <= out <= pi + (1 * (3 * 2)) + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_softplus_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (pi + (1 * (3 * 2)) - 2 * D.epsilon() <= D.addcmul(pi, D.to_float(3), D.to_float(2), value=1) <= pi + (1 * (3 * 2)) + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_equal(ffmt):
    D.set_float_fmt(ffmt)
    a = D.array([[[1.0, 2.0, 3.0], [2.0, 5.0, 8.0]], [[1.0, -2.0, 3.0], [12.0, 5.0, -8.0]]])
    assert(D.equal(a, a))


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_remainder_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.14159265358979324 - 2 * D.epsilon() <= D.remainder(pi, 3) <= 0.14159265358979324 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_ceil_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.ceil(pi) == 4)


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_floor_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.floor(pi) == 3)


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_round_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.round(pi) == 3)


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_fmod_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.1415926535897931 - 2 * D.epsilon() <= D.fmod(pi, 2) <= 1.1415926535897931 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_clip_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.clip(pi, 1, 2) == 2)


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_sign_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.sign(pi) == 1)


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_trunc_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (D.trunc(pi) == 3)


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_digamma_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.9772133079420067 - 2 * D.epsilon() <= D.digamma(pi) <= 0.9772133079420067 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_erfinv_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.4769362762044699 - 2 * D.epsilon() <= D.erfinv(D.to_float(0.5)) <= 0.4769362762044699 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_mvlgamma_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (1.7891115385869942 - 2 * D.epsilon() <= D.mvlgamma(pi, 2) <= 1.7891115385869942 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_frac_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert (0.141592653589793238 - 2 * D.epsilon() <= D.frac(pi) <= 0.141592653589793238 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_frac_within_tolerance_out(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    out = D.copy(pi)
    D.frac(pi, out=out)
    assert (0.141592653589793238 - 2 * D.epsilon() <= out <= 0.141592653589793238 + 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_einsum_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    assert (-2 * D.epsilon() <= D.einsum("nm->", D.array([[1.0, 2.0], [-2.0, -1.0]])) <= 2 * D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_dist_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    arr1 = D.array([2.0, 0.0])
    arr2 = D.array([0.0, 2.0])
    assert (2.8284271247461900976 - 2*D.epsilon() <= D.dist(arr1, arr2, ord=2) <= 2.8284271247461900976 + 2*D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_softplus_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert(3.18389890758499587775 - 2*D.epsilon() <= D.softplus(pi) <= 3.18389890758499587775 + 2*D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_softplus_within_tolerance_out(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    out = D.copy(pi)
    D.softplus(pi, out=out)
    assert(3.18389890758499587775 - 2*D.epsilon() <= out <= 3.18389890758499587775 + 2*D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_square_within_tolerance(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    assert(9.8696044010893586188 - 2*D.epsilon() <= D.square(pi) <= 9.8696044010893586188 + 2*D.epsilon())


@pytest.mark.parametrize('ffmt', list(i for i in D.available_float_fmt() if not i.startswith('gdual')))
def test_square_within_tolerance_out(ffmt):
    D.set_float_fmt(ffmt)
    pi = D.to_float(D.pi)
    out = D.copy(pi)
    D.square(pi, out=out)
    assert(9.8696044010893586188 - 2*D.epsilon() <= out <= 9.8696044010893586188 + 2*D.epsilon())


def test_logical_not():
    a = D.array([True, False, False, True], dtype=D.bool)
    ref = D.array([False, True, True, False], dtype=D.bool)
    assert (D.all(D.logical_not(a) == ref))


def test_logical_not_out():
    a = D.array([True, False, False, True], dtype=D.bool)
    ref = D.array([False, True, True, False], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    D.logical_not(a, out=out)
    assert (D.all(out == ref))


def test_logical_not_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    ref = D.array([False, False, False, False], dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    assert (D.all(D.logical_not(a, where=where)[where] == ref[where]))


def test_logical_not_out_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    ref = D.array([False, False, False, False], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    D.logical_not(a, out=out, where=where)
    assert (D.all(out[where] == ref[where]))


def test_logical_or():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, True], dtype=D.bool)
    assert (D.all(D.logical_or(a, b) == ref))


def test_logical_or_out():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, True], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    D.logical_or(a, b, out=out)
    assert (D.all(out == ref))


def test_logical_or_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, True], dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    assert (D.all(D.logical_or(a, b, where=where)[where] == ref[where]))


def test_logical_or_out_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, True], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    D.logical_or(a, b, out=out, where=where)
    assert (D.all(out[where] == ref[where]))


def test_logical_and():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([False, False, False, True], dtype=D.bool)
    assert (D.all(D.logical_and(a, b) == ref))


def test_logical_and_out():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([False, False, False, True], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    D.logical_and(a, b, out=out)
    assert (D.all(out == ref))


def test_logical_and_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([False, False, False, True], dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    assert (D.all(D.logical_and(a, b, where=where)[where] == ref[where]))


def test_logical_and_out_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([False, False, False, True], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    D.logical_and(a, b, out=out, where=where)
    assert (D.all(out[where] == ref[where]))


def test_logical_xor():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, False], dtype=D.bool)
    assert (D.all(D.logical_xor(a, b) == ref))


def test_logical_xor_out():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, False], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    D.logical_xor(a, b, out=out)
    assert (D.all(out == ref))


def test_logical_xor_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, False], dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    assert (D.all(D.logical_xor(a, b, where=where)[where] == ref[where]))


def test_logical_xor_out_where():
    a = D.array([True, False, False, True], dtype=D.bool)
    b = D.array([False, False, True, True], dtype=D.bool)
    ref = D.array([True, False, True, False], dtype=D.bool)
    out = D.zeros_like(a, dtype=D.bool)
    where = D.array([True, False, False, True], dtype=D.bool)
    D.logical_xor(a, b, out=out, where=where)
    assert (D.all(out[where] == ref[where]))



@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_linspace():
    ref_lin = np.linspace(1.0, 10.0, num=100)
    test_lin = D.linspace(1.0, 10.0, num=100)
    assert (np.all(np.abs(ref_lin - test_lin.cpu().numpy())/ref_lin <= 10 * D.epsilon()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_logspace():
    ref_lin = np.logspace(-10.0, 10.0, num=100)
    test_lin = D.logspace(-10.0, 10.0, num=100)
    assert (np.all(np.abs(ref_lin - test_lin.cpu().numpy())/ref_lin <= 10 * D.epsilon()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_ravel():
    a = np.array([[1.0, 2.0, 3.0]])
    a_torch = D.array(a)
    assert (np.all(np.ravel(a) == D.ravel(a_torch).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_flatten():
    a = np.array([[1.0, 2.0, 3.0]])
    a_torch = D.array(a)
    assert (np.all(np.ravel(a) == D.flatten(a_torch).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_append():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert (np.all(np.append(a, b) == D.append(a_torch, b_torch).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_append_axis_none():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0]])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert (np.all(np.append(a, b, axis=None) == D.append(a_torch, b_torch, axis=None).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_concatenate():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert (np.all(np.concatenate([a, b, a, b]) ==
                   D.concatenate([a_torch, b_torch, a_torch, b_torch]).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_concatenate_with_axis():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0]])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert (np.all(np.concatenate([a, b, a, b], axis=1) ==
                   D.concatenate([a_torch, b_torch, a_torch, b_torch], axis=1).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_concatenate_with_none_axis():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0]])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert (np.all(np.concatenate([a, b, a, b], axis=None) ==
                   D.concatenate([a_torch, b_torch, a_torch, b_torch], axis=None).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_max():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 3.0, -1.0]])
    a_torch = D.array(a)
    assert (np.all(np.max(a, axis=None) == D.max(a_torch, axis=None).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_max_with_axis():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 3.0, -1.0]])
    a_torch = D.array(a)
    assert (np.all(np.max(a, axis=1) == D.max(a_torch, axis=1).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_min():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 3.0, -1.0]])
    a_torch = D.array(a)
    assert (np.all(np.min(a, axis=None) == D.min(a_torch, axis=None).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_torch_min_with_axis():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 3.0, -1.0]])
    a_torch = D.array(a)
    assert (np.all(np.min(a, axis=1) == D.min(a_torch, axis=1).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_to_numpy():
    a = np.array([1.0, 2.0, 3.0])
    a_torch = D.array(a)
    assert (np.all(np.stack([a, a]) == D.to_numpy([a_torch, a_torch])))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_asarray():
    a = np.array([1.0, 2.0, 3.0])
    a_torch = D.array(a)
    assert (D.all(a_torch == D.asarray(a)))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_shape():
    a = np.array([1.0, 2.0, 3.0])
    a_torch = D.array(a)
    assert (D.shape(a_torch) == (3,))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_shape_has_no_shape():
    a = [1.0, 2.0, 3.0]
    assert (D.shape(a) == (3,))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_eye():
    assert (np.all(np.eye(5) == D.eye(5).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_eye_asymm():
    assert (np.all(np.eye(5, 2) == D.eye(5, 2).cpu().numpy()))
    assert (np.all(np.eye(5, 20) == D.eye(5, 20).cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_eye_with_out():
    out = D.zeros((5, 5))
    D.eye(5, out=out)
    assert (np.all(np.eye(5) == out.cpu().numpy()))


@pytest.mark.skipif(D.backend() == 'numpy', reason="Numpy is Reference")
def test_eye_asymm_with_out():
    out = D.zeros((5, 2))
    D.eye(5, 2, out=out)
    assert (np.all(np.eye(5, 2) == out.cpu().numpy()))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_subtraction():
    a = D.array([1.0])
    assert (D.all(D.sub(a, a) == D.array([0.0])))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_subtraction_with_out():
    a = D.array([1.0])
    out = D.zeros((1,))
    D.sub(a, a, out=out)
    assert (D.all(out == D.array([0.0])))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_wrong_nu():
    with pytest.raises(ValueError):
        a = D.array([1.0, 1.0], requires_grad=True)
        D.jacobian(a, a, nu=-1)


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_nu_zero():
    a = D.array([1.0, 1.0], requires_grad=True)
    assert (D.all(a == D.jacobian(a, a, nu=0)))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_no_grad():
    a = D.array([1.0, 1.0], requires_grad=False)
    assert (D.all(D.jacobian(a, a) == D.array([[0.0, 0.0], [0.0, 0.0]])))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_no_grad_batched():
    a = D.array([[1.0, 1.0], [1.0, 1.0]], requires_grad=False)
    assert (D.all(D.jacobian(a, a, batch_mode=True) == D.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_batched():
    a = D.array([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    b = a * D.array([[1.0, 1.0], [2.0, 2.0]])
    assert (D.all(D.jacobian(b, a, batch_mode=True) == D.array([[[1.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [0.0, 2.0]]])))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_unbatched():
    a = D.array([1.0, 1.0], requires_grad=True)
    assert (D.all(D.jacobian(a, a, batch_mode=False) == D.array([[1.0, 0.0], [0.0, 1.0]])))


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
def test_gradients_higher_nu():
    a = D.array([1.0], requires_grad=True)
    b = a * a
    assert (D.all(D.jacobian(b, a, nu=2, batch_mode=False) == D.array([2.0])))
    

@pytest.mark.skipif("gdual_double" not in D.available_float_fmt(), reason="Can't test dispatch without pyaudi overload")
def test_matrix_inv():
    A  = D.array([
        [-1.0,  3/2],
        [ 1.0, -1.0],
    ], dtype=D.float64)
    Ainv = D.matrix_inv(A)
    assert (D.max(D.abs(D.to_float(Ainv@A - D.eye(2)))) <= 8*D.epsilon())
    with pytest.raises(np.linalg.LinAlgError):
        D.matrix_inv(D.zeros((2,3), dtype=D.float64))
    with pytest.raises(np.linalg.LinAlgError):
        D.matrix_inv(D.zeros((5,2,3), dtype=D.float64))

class PyAudiTestCase:
    @pytest.mark.skipif(D.backend() != 'numpy' or 'gdual_double' not in D.available_float_fmt(), reason="PyAudi Tests")
    def test_gdual_double(self):
        x1 = D.gdual_double(-0.5, 'x', 5)
        x2 = D.gdual_double(0.5, 'y', 5)
        self.do(x1, x2)

    @pytest.mark.skipif(D.backend() != 'numpy' or 'gdual_double' not in D.available_float_fmt(), reason="PyAudi Tests")
    def test_gdual_vdouble(self):
        x1 = D.gdual_vdouble([-0.5, -0.5], 'x', 5)
        x2 = D.gdual_vdouble([0.5, 0.5], 'y', 5)
        self.do(x1, x2)
        
    @pytest.mark.skipif(D.backend() != 'numpy' or 'gdual_double' not in D.available_float_fmt(), reason="PyAudi Tests")
    def test_gdual_double_matrix(self):
        A  = D.array([
            [D.gdual_double(-1.0, 'a11', 5), D.gdual_double( 3/2, 'a12', 5)],
            [D.gdual_double( 1.0, 'a21', 5), D.gdual_double(-1.0, 'a22', 5)],
        ])
        self.do_matrix(A)
        
    def do(self, x1, x2):
        pass
    
    def do_matrix(self, A):
        pass


class TestPyAudiFloat(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.__float__()
        assert (res == x1.constant_cf)


class TestPyAudiAbs(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = abs(x1)
        assert (res == pd.abs(x1))


class TestPyAudiSqrt(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x2.sqrt()
        assert (res == pd.sqrt(x2))


class TestPyAudiExp(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.exp()
        assert (res == pd.exp(x1))


class TestPyAudiExpm1(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.expm1()
        assert (res == pd.exp(x1) - 1.0)


class TestPyAudiLog(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x2.log()
        assert (res == pd.log(x2))


class TestPyAudiLog10(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x2.log10()
        assert (res == pd.log(x2)/np.log(10.0))


class TestPyAudiLog1p(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x2.log1p()
        assert (res == pd.log(x2 + 1.0))


class TestPyAudiLog2(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x2.log2()
        assert (res == pd.log(x2)/np.log(2.0))


class TestPyAudiCos(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.cos()
        assert (res == pd.cos(x1))


class TestPyAudiSin(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.sin()
        assert (res == pd.sin(x1))


class TestPyAudiTan(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.tan()
        assert (res == pd.tan(x1))


class TestPyAudiArcCos(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.arccos()
        assert (res == pd.acos(x1))


class TestPyAudiArcSin(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.arcsin()
        assert (res == pd.asin(x1))


class TestPyAudiArcTan(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.arctan()
        assert (res == pd.atan(x1))


class TestPyAudiArcTan2(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.arctan2(x2)
        if isinstance(x2, D.gdual_vdouble):
            fres = pd.atan(x1/x2) + (D.gdual_vdouble(list(map(lambda x: (x < 0) * np.pi, x2.constant_cf))))
        else:
            fres = pd.atan(x1/x2) + (float(x2) < 0) * np.pi
        assert (res == fres)


class TestPyAudiCosh(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.cosh()
        assert (res == pd.cosh(x1))


class TestPyAudiSinh(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.sinh()
        assert (res == pd.sinh(x1))


class TestPyAudiTanh(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.tanh()
        assert (res == pd.tanh(x1))


class TestPyAudiErf(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.erf()
        assert (res == pd.erf(x1))


class TestPyAudiErfc(PyAudiTestCase):
    def do(self, x1, x2):
        import pyaudi as pd
        res = x1.erfc()
        assert (res == 1.0 - pd.erf(x1))

class TestPyAudiMatrixInv(PyAudiTestCase):
    def do_matrix(self, A):
        Ainv = D.matrix_inv(A)
        assert (D.max(D.abs(D.to_float(Ainv@A - D.eye(A.shape[0])))) <= 8*D.epsilon())
        
        with pytest.raises(np.linalg.LinAlgError):
            self.do_matrix(D.zeros((2,3)))
        with pytest.raises(np.linalg.LinAlgError):
            self.do_matrix(D.zeros((5,2,3)))