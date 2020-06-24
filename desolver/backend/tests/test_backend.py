import os
import numpy as np
import nose
import desolver as de
import desolver.backend as D

def test_backend():
    try:
        import numpy as np
        import scipy
        
        if "DES_BACKEND" in os.environ:
            assert(D.backend() == os.environ['DES_BACKEND'])
        
        if D.backend() not in ['torch']:
            # Default datatype test
            for i in D.available_float_fmt():
                D.set_float_fmt(i)
                assert(D.array(1.0).dtype  == D.float_fmts[D.float_fmt()])

        expected_eps = {'float16': 5e-3, 'float32': 5e-7, 'float64': 5e-16, 'gdual_double': 5e-16, 'gdual_vdouble': 5e-16, 'gdual_real128': 5e-16}
        test_array   = np.array([1], dtype=np.int64)
        # Test Function Evals
        for i in D.available_float_fmt():
            D.set_float_fmt(i)
            assert(D.float_fmt() == str(i))
            assert(D.epsilon() == expected_eps[str(i)])
            assert(isinstance(D.available_float_fmt(), list))
            if not i.startswith('gdual'):
                assert(D.cast_to_float_fmt(test_array).dtype == str(i))
            
            arr1 = D.array([[2.0, 1.0],[1.0, 0.0]])
            arr2 = D.array([[1.0, 1.0],[-1.0, 1.0]])
            
            if not i.startswith('gdual'):
                arr3 = D.contract_first_ndims(arr1, arr2, 1)
                arr4 = D.contract_first_ndims(arr1, arr2, 2)
            
                true_arr3 = D.array([1.0, 1.0])
                true_arr4 = D.array(2.)

                assert(D.norm(arr3 - true_arr3) <= 2 * D.epsilon())
                assert(D.norm(arr4 - true_arr4) <= 2 * D.epsilon())
            
            de.utilities.warning("Testing float format {}".format(D.float_fmt()))
            
            pi = D.to_float(D.pi)
            
            assert(np.pi - 2*D.epsilon()          <= pi                        <= np.pi + 2*D.epsilon())
            assert(np.e - 2*D.epsilon()           <= D.to_float(D.e)           <= np.e + 2*D.epsilon())
            assert(np.euler_gamma - 2*D.epsilon() <= D.to_float(D.euler_gamma) <= np.euler_gamma + 2*D.epsilon())
            
            assert(-2*D.epsilon() <= D.sin(pi) <= 2*D.epsilon())
            assert(-2*D.epsilon() <= D.cos(pi)+1 <= 2*D.epsilon())
            assert(-2*D.epsilon() <= D.tan(pi) <= 2*D.epsilon())

            assert(D.asin(D.to_float(1)) == pi/2)
            assert(D.acos(D.to_float(1)) == 0)
            assert(D.atan(D.to_float(1)) == pi/4)
            assert(D.atan2(D.to_float(1), D.to_float(1)) == pi/4)

            assert(D.sinh(pi)        == np.sinh(pi))
            assert(D.cosh(pi)        == np.cosh(pi))
            assert(D.tanh(pi)        == np.tanh(pi))

            assert(-3.141592653589793 - 2*D.epsilon()  <= D.neg(pi)   <= -3.141592653589793 + 2*D.epsilon())
            assert(31.00627668029982 - 10*D.epsilon()  <= D.pow(pi,3) <= 31.00627668029982 + 10*D.epsilon())
            assert(3.141592653589793 - 2*D.epsilon()   <= D.abs(pi)   <= 3.141592653589793 + 2*D.epsilon())
            assert(1.77245385090551603 - 2*D.epsilon() <= D.sqrt(pi)  <= 1.77245385090551603 + 2*D.epsilon())
            assert(23.1406926327792690 - 10*D.epsilon()<= D.exp(pi)   <= 23.1406926327792690 + 10*D.epsilon())
            assert(22.1406926327792690 - 10*D.epsilon()<= D.expm1(pi) <= 22.1406926327792690 + 10*D.epsilon())
            assert(1.14472988584940017 - 2*D.epsilon() <= D.log(pi)   <= 1.14472988584940017 + 2*D.epsilon())
            assert(1.14472988584940017 - 2*D.epsilon() <= D.log(pi)   <= 1.14472988584940017 + 2*D.epsilon())
            assert(0.49714987269413385 - 2*D.epsilon() <= D.log10(pi) <= 0.49714987269413385 + 2*D.epsilon())
            assert(1.42108041279429263 - 2*D.epsilon() <= D.log1p(pi) <= 1.42108041279429263 + 2*D.epsilon())
            assert(1.65149612947231880 - 2*D.epsilon() <= D.log2(pi)  <= 1.65149612947231880 + 2*D.epsilon())

            assert(4.14159265358979324 - 2*D.epsilon() <= D.add(pi,1) <= 4.14159265358979324 + 2*D.epsilon())
            assert(2.14159265358979324 - 2*D.epsilon() <= D.sub(pi,1) <= 2.14159265358979324 + 2*D.epsilon())
            assert(D.div(pi,1)       == pi)
            assert(D.mul(pi,1)       == pi)

            assert(0.31830988618379067 - 2*D.epsilon() <= D.reciprocal(pi)  <= 0.31830988618379067 + 2*D.epsilon())
            
            if not i.startswith('gdual'):
                assert(0.14159265358979324 - 2*D.epsilon() <= D.remainder(pi,3) <= 0.14159265358979324 + 2*D.epsilon())
                assert(D.ceil(pi)        == 4)
                assert(D.floor(pi)       == 3)
                assert(D.round(pi)       == 3)
                assert(1.1415926535897931 - 2*D.epsilon()  <= D.fmod(pi,2) <= 1.1415926535897931 + 2*D.epsilon())

                assert(D.clip(pi,1,2)    == 2)
                assert(D.sign(pi)        == 1)
                assert(D.trunc(pi)       == 3)

                assert(0.9772133079420067 - 2*D.epsilon()    <= D.digamma(pi)             <= 0.9772133079420067 + 2*D.epsilon())
                assert(0.4769362762044699 - 2*D.epsilon()    <= D.erfinv(D.to_float(0.5)) <= 0.4769362762044699 + 2*D.epsilon())
                assert(1.7891115385869942 - 2*D.epsilon()    <= D.mvlgamma(pi, 2)         <= 1.7891115385869942 + 2*D.epsilon())
                assert(D.frac(pi)            == pi - 3)
                
            assert(0.9999911238536324 - 2*D.epsilon()    <= D.erf(pi)                 <= 0.9999911238536324 + 2*D.epsilon())
            assert(8.8761463676416054e-6 - 2*D.epsilon() <= D.erfc(pi)                <= 8.8761463676416054e-6 + 2*D.epsilon())
            
            assert(0.9585761678336372 - 2*D.epsilon()    <= D.sigmoid(pi)             <= 0.9585761678336372 + 2*D.epsilon())

            assert(0.5641895835477563 - 2*D.epsilon()    <= D.rsqrt(pi)               <= 0.5641895835477563 + 2*D.epsilon())
            assert(pi + 0.5 - 2*D.epsilon()              <= D.lerp(pi,pi+1,0.5)       <= pi + 0.5 + 2*D.epsilon())
            
            
            assert(D.addcdiv(pi,1,D.to_float(3),D.to_float(2))   == pi + (1 * (3 / 2)))
            assert(D.addcmul(pi,1,D.to_float(3),D.to_float(2))   == pi + (1 * (3 * 2)))
            
            if not i.startswith('gdual'):
                assert(-2*D.epsilon() <= D.einsum("nm->", D.array([[1.0, 2.0], [-2.0, -1.0]])) <= 2*D.epsilon())
    except:
        print("{} Backend Test Failed".format(D.backend()))
        raise
    print("{} Backend Test Succeeded".format(D.backend())) 
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_subtraction():
    a = D.array([1.0])
    assert(D.all(D.sub(a, a) == D.array([0.0])))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_subtraction_with_out():
    a = D.array([1.0])
    out = D.zeros((1,))
    D.sub(a, a, out=out)
    assert(D.all(out == D.array([0.0])))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_logspace():
    assert(np.all(np.abs(np.logspace(-10.0, 10.0) - D.logspace(-10.0, 10.0).cpu().numpy()) <= 2*D.epsilon()))

@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
@nose.tools.raises(ValueError)
def test_gradients_wrong_nu():
    a = D.array([1.0,1.0], requires_grad=True)
    D.jacobian(a, a, nu=-1)
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_gradients_nu_zero():
    a = D.array([1.0,1.0], requires_grad=True)
    assert(D.all(a == D.jacobian(a, a, nu=0)))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_gradients_no_grad():
    a = D.array([1.0,1.0], requires_grad=False)
    assert(D.all(D.jacobian(a, a) == D.array([[0.0,0.0],[0.0,0.0]])))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_gradients_no_grad_batched():
    a = D.array([[1.0,1.0], [1.0,1.0]], requires_grad=False)
    assert(D.all(D.jacobian(a, a, batch_mode=True) == D.array([[[0.0,0.0],[0.0,0.0]],[[0.0,0.0],[0.0,0.0]]])))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_gradients_batched():
    a = D.array([[1.0,1.0], [1.0,1.0]], requires_grad=True)
    b = a * D.array([[1.0,1.0], [2.0,2.0]])
    assert(D.all(D.jacobian(b, a, batch_mode=True) == D.array([[[1.0,0.0],[2.0,0.0]],[[0.0,1.0],[0.0,2.0]]])))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_gradients_unbatched():
    a = D.array([1.0,1.0], requires_grad=True)
    assert(D.all(D.jacobian(a, a, batch_mode=False) == D.array([[1.0,0.0],[0.0,1.0]])))
    
@np.testing.dec.skipif(D.backend() != 'torch', "PyTorch Unavailable")
def test_gradients_higher_nu():
    a = D.array([1.0], requires_grad=True)
    b = a*a
    assert(D.all(D.jacobian(b, a, nu=2, batch_mode=False) == D.array([2.0])))

@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_torch_ravel():
    a = np.array([[1.0, 2.0, 3.0]])
    a_torch = D.array(a)
    assert(np.all(np.ravel(a) == D.ravel(a_torch).cpu().numpy()))

@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_torch_append():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert(np.all(np.append(a, b) == D.append(a_torch, b_torch).cpu().numpy()))
    
@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_torch_append_axis_none():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0]])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert(np.all(np.append(a, b, axis=None) == D.append(a_torch, b_torch, axis=None).cpu().numpy()))

@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_torch_concatenate():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert(np.all(np.concatenate([a, b, a, b]) == D.concatenate([a_torch, b_torch, a_torch, b_torch]).cpu().numpy()))

@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_torch_concatenate_with_axis():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[1.0]])
    a_torch = D.array(a)
    b_torch = D.array(b)
    assert(np.all(np.concatenate([a, b, a, b], axis=1) == D.concatenate([a_torch, b_torch, a_torch, b_torch], axis=1).cpu().numpy()))
    
@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_eye():
    assert(np.all(np.eye(5) == D.eye(5).cpu().numpy()))
    
@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_eye_asymm():
    assert(np.all(np.eye(5,2) == D.eye(5,2).cpu().numpy()))
    assert(np.all(np.eye(5,20) == D.eye(5,20).cpu().numpy()))
    
@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_eye_with_out():
    out = D.zeros((5,5))
    D.eye(5, out=out)
    assert(np.all(np.eye(5) == out.cpu().numpy()))
    
@np.testing.dec.skipif(D.backend() == 'numpy', "Numpy is Reference")
def test_eye_asymm_with_out():
    out = D.zeros((5,2))
    D.eye(5,2, out=out)
    assert(np.all(np.eye(5,2) == out.cpu().numpy()))

if __name__ == "__main__":
    np.testing.run_module_suite()
















