import os
os.environ['DES_BACKEND'] = 'torch'
import desolver as de
import desolver.backend as D
import numpy as np

def test_brentsroot():
    for fmt in D.available_float_fmt():
        print("Set dtype to:", fmt)
        D.set_float_fmt(fmt)
        for _ in range(10):
            ac_prod = D.array(np.random.uniform(0.9, 1.1))
            a = D.array(np.random.uniform(-1, 1))
            c = ac_prod / a
            b = D.sqrt(0.01 + 4*ac_prod)

            gt_root = -b / (2*a) - 0.1 / (2*a)

            ub  = -b / (2*a)
            lb  = -b / (2*a) - 1.0 / (2*a)

            fun = lambda x: a*x**2 + b*x + c

            assert(D.to_numpy(D.to_float(D.abs(fun(gt_root)))) <= 8*D.epsilon())

            root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], 4*D.epsilon(), verbose=True)

            assert(success)
            assert(np.allclose(D.to_numpy(D.to_float(gt_root)), D.to_numpy(D.to_float(root)), 8*D.epsilon(), 8*D.epsilon()))
            assert(D.to_numpy(D.to_float(D.abs(fun(root)))) <= 8*D.epsilon())

def test_brentsrootvec():
    for fmt in D.available_float_fmt():
        print("Set dtype to:", fmt)
        D.set_float_fmt(fmt)
        for _ in range(10):
            slope_list     = D.array(np.random.uniform(-1, 1, size=25))
            intercept_list = D.array(np.random.uniform(-1, 1, size=25))

            gt_root_list = -intercept_list/slope_list

            fun_list = [(lambda m,b: lambda x: m*x + b)(m,b) for m,b in zip(slope_list, intercept_list)]

            assert(all(map((lambda i: D.to_numpy(D.to_float(D.abs(i))) <= 8*D.epsilon()), map((lambda x: x[0](x[1])), zip(fun_list, gt_root_list)))))

            root_list, success = de.utilities.optimizer.brentsrootvec(fun_list, [D.min(gt_root_list) - 1., D.max(gt_root_list) + 1.], 4*D.epsilon(), verbose=True)

            assert(np.all(D.to_numpy(success)))
            assert(np.allclose(D.to_numpy(D.to_float(gt_root_list)), D.to_numpy(D.to_float(root_list)), 8*D.epsilon(), 8*D.epsilon()))

            assert(all(map((lambda i: D.to_numpy(D.to_float(D.abs(i))) <= 8*D.epsilon()), map((lambda x: x[0](x[1])), zip(fun_list, root_list)))))

if __name__ == "__main__":
    np.testing.run_module_suite()
