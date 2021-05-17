import time
import sys
import numpy
import warnings
from .. import backend as D

__all__ = [
    'JacobianWrapper',
    'convert_suffix',
    'warning',
    'search_bisection',
    'search_bisection_vec',
    'BlockTimer'
]


class JacobianWrapper(object):
    """
    A wrapper class that uses Richardson Extrapolation and 4th order finite differences to compute the jacobian of a given callable function.
    
    The Jacobian is computed using up to 16 richardson iterations which translates to a maximum order of 4+16 for the gradients with respect to Δx as Δx->0. The evalaution is adaptive so within the given tolerances, the evaluation will exit early in order to minimise computation so gradients can be calculated with controllable precision.
    
    Attributes
    ----------
    rhs : Callable of the form f(x)->D.array of arbitrary shape
        A callable function that 
        
    base_order : int
        The order of the underlying finite difference gradient estimate, this can be evaluated at different step sizes using the `estimate` method.
        
    richardson_iter : int
        The maximum number of richardson iterations to use for estimating gradients. In most cases, less than 16 should be enough, and if you start needing more than 16, it might be worth considering that finite differences are inappropriate.
        
    order : int
        Maximum order of the gradient estimate
    
    adaptive : bool
        Whether to use adaptive evaluation of the richardson extrapolation or to run for the full 16 iterations.
        
    atol, rtol : float
        Absolute and relative tolerances for the adaptive gradient extrapolation
        
    flat : bool
        If set to True, the gradients will be returned as a ravelled array (ie. jacobian.shape = (-1,))
        If set to False, the shape will be (*x.shape, *f(x).shape). When x and f(x) are vector valued of length n and m respectively, the gradient will have the shape (n,m) as expected of the Jacobian of f(x).
    
    """

    def __init__(self, rhs, base_order=2, richardson_iter=None, adaptive=True, flat=False, atol=None, rtol=None):
        self.rhs = rhs
        self.base_order = base_order
        if richardson_iter is None:
            self.richardson_iter = 16 - base_order if base_order < 16 else base_order + 1
        else:
            self.richardson_iter = richardson_iter
        self.order = self.base_order + self.richardson_iter
        self.adaptive = adaptive
        self.atol = 4 * D.epsilon() if atol is None else atol
        self.rtol = 4 * D.epsilon() if rtol is None else rtol
        self.flat = flat
        self.nodal_points, self.weights = self.finite_difference_weights(self.base_order, order=1)
        self.nodal_points = self.nodal_points[D.abs(self.weights) > 16 * D.epsilon()]
        self.weights = self.weights[D.abs(self.weights) > 16 * D.epsilon()]

    @staticmethod
    def finite_difference_weights(number_of_nodes, order=1):
        cur_ffmt = D.float_fmt()
        D.set_float_fmt('float64')
        nodal_points = D.linspace(-1, 1, number_of_nodes)
        weight_matrix = D.stack(
            [D.pow(D.to_type(D.asarray(nodal_points), D.float64), i) for i in range(len(nodal_points))])
        b_vector = D.zeros((len(nodal_points),), dtype=D.float64)
        b_vector[order] = 1.0
        if D.backend() == 'torch':
            b_vector = b_vector[:, None]
        weights = D.solve_linear_system(weight_matrix, b_vector)
        if D.backend() == 'torch':
            weights = weights[:, 0]
        D.set_float_fmt(cur_ffmt)
        return nodal_points, weights

    def estimate(self, y, dy=None, **kwargs):
        if dy is None:
            dy = D.epsilon() ** 0.5
        unravelled_y = D.reshape(y, (-1,))
        dy_val = self.rhs(y, **kwargs)
        unravelled_dy = D.reshape(dy_val, (-1,))
        jacobian_y = D.zeros((*D.shape(unravelled_dy), *D.shape(unravelled_y)), dtype=unravelled_dy.dtype)
        y_msk = D.zeros_like(unravelled_y)
        if D.backend() == 'torch':
            jacobian_y = jacobian_y.to(dy_val).to(y.device)
        for idx, val in enumerate(unravelled_y):
            y_msk[idx - 1] = 0.0
            y_msk[idx] = 1.0
            dy_cur = dy
            if not self.adaptive and (D.abs(val) > 1.0 or dy_cur > D.abs(val) > 0.0):
                dy_cur = dy_cur * val

            for A, w in zip(self.nodal_points, self.weights):
                y_jac = unravelled_y + A * dy_cur * y_msk
                jacobian_y[:, idx] = jacobian_y[:, idx] + w * D.reshape(
                    self.rhs(D.reshape(y_jac, D.shape(y)), **kwargs), (-1,))

            jacobian_y[:, idx] = jacobian_y[:, idx] / dy_cur

        if self.flat:
            if D.shape(jacobian_y) == (1, 1):
                return jacobian_y[0, 0]
            return jacobian_y
        else:
            return jacobian_y.reshape((*D.shape(dy_val), *D.shape(y)))

    def richardson(self, y, dy=0.5, factor=4.0, **kwargs):
        A = [[self.estimate(y, dy=dy * (factor ** -m), **kwargs)] for m in range(self.richardson_iter)]
        denom = factor ** self.base_order
        for m in range(1, self.richardson_iter):
            for n in range(1, m):
                A[m].append(A[m][n - 1] + (A[m][n - 1] - A[m - 1][n - 1]) / (denom ** n - 1))
        return A[-1][-1]

    def adaptive_richardson(self, y, dy=0.5, factor=4, **kwargs):
        A = [[self.estimate(y, dy=dy, **kwargs)]]
        if self.richardson_iter == 1:
            return A[0][0]
        factor = 1.0 * factor
        denom = factor ** self.base_order
        prev_error = numpy.inf
        for m in range(1, self.richardson_iter):
            A.append([self.estimate(y, dy=dy * (factor ** (-m)), **kwargs)])
            for n in range(1, m + 1):
                A[m].append(A[m][n - 1] + (A[m][n - 1] - A[m - 1][n - 1]) / (denom ** n - 1))
            if m >= 3:
                prev_error, t_conv = self.check_converged(A[m][m], A[m][m] - A[m - 1][m - 1], prev_error)
                if t_conv:
                    self.order = self.base_order + m
                    break
        return A[-2][-1]

    def check_converged(self, initial_state, diff, prev_error):
        err_estimate = D.max(D.abs(D.to_float(diff)))
        relerr = D.max(D.to_float(self.atol + self.rtol * D.abs(initial_state)))
        if err_estimate > relerr and err_estimate < prev_error:
            return err_estimate, False
        else:
            return err_estimate, True

    def __call__(self, y, **kwargs):
        if self.richardson_iter > 0:
            if self.adaptive:
                out = self.adaptive_richardson(y, **kwargs)
            else:
                out = self.richardson(y, **kwargs)
        else:
            out = self.estimate(y, **kwargs)
        return out


def convert_suffix(value, suffixes=('d', 'h', 'm', 's'), ratios=(24, 60, 60), delimiter=':'):
    """Converts a base value into a human readable format with the given suffixes and ratios.

    Parameters
    ----------
    value : int or float
        value to be converted
    suffixes : list of str
        suffixes for each subdivision (eg. ['days', 'hours', 'minutes', 'seconds'])
    ratios : list of int
        the relative period of each subdivision (eg. 24 hours in 1 day, 60 minutes in 1 hour, 
        60 seconds in 1 minute -> [24, 60, 60])
    delimiter : str
        string to use between each subdivision

    Returns
    -------
    str
        returns string with the subdivision values and suffixes joined together

    Examples
    --------
    
    >>> convert_suffix(3661, suffixes=['d', 'h', 'm', 's'], ratios=[24, 60, 60], delimiter=':')
    '0d:1h:1m1.00s'
    
    """
    tValue = value
    outputValues = []
    for i in ratios[::-1]:
        outputValues.append(int(tValue % i))
        tValue = (tValue - tValue % i) // i
    outputValues.append(tValue)
    ret_string = delimiter.join(["{}{}".format(int(i[0]), i[1]) for i in zip(outputValues[::-1][:-1], suffixes[:-1])])
    ret_string = ret_string + "{:.2f}{}".format(outputValues[0], suffixes[-1])
    return ret_string


def warning(message, category=Warning):
    """Convenience function for printing to sys.stderr. 

    Parameters
    ----------
    message  : str
        warning message
    category : warning category
        type of warning message. eg. DeprecationWarning
        
    Examples
    --------
    
    >>> warning("Things have failed...", warning.Warning)
    Things have failed...
    
    """
    warnings.warn(message, category=category)


def search_bisection(array, val):
    """Finds the index of the nearest value to val in array. Uses the bisection method.

    Parameters
    ----------
    array : list of numeric values
        list to search, assumes the list is sorted (will not work if it isn't sorted!)
    val : numeric
        numeric value to find the nearest value in the array.

    Returns
    -------
    int
        returns the index of the position in the array with the value closest to val

    Examples
    --------
    
    >>> list_to_search = [1,2,3,4,5]
    >>> val_to_find    = 2.5
    >>> idx = search_bisection(list_to_search, val_to_find)
    >>> idx, list_to_search[idx]
    (1, 2)
    
    """

    jlower = 0
    jupper = len(array) - 1

    if val <= array[jlower]:
        return jlower
    elif val >= array[jupper]:
        return jupper

    while (jupper - jlower) > 1:
        jmid = (jupper + jlower) // 2
        if (val >= array[jmid]):
            jlower = jmid
        else:
            jupper = jmid

    return jlower


def search_bisection_vec(array, val):
    """Finds the indices of the nearest values in array to each val. Uses the bisection method.

    Parameters
    ----------
    array : array of numeric values
        list to search, assumes the list is sorted (will not work if it isn't sorted!)
    val : array of numeric values
        numeric values to find the nearest value in array.

    Returns
    -------
    array(int)
        returns the indices of the positions in the array with the value closest to val

    Examples
    --------
    
    >>> list_to_search = [1,2,3,4,5]
    >>> val_to_find    = [1.5,3.5]
    >>> idx = search_bisection(list_to_search, val_to_find)
    >>> idx, list_to_search[idx]
    ([0,2], [1,3])
    
    """

    val = D.asarray(val)
    array = D.asarray(array)
    jlower = D.zeros_like(val, dtype=D.int64)
    jupper = D.ones_like(val, dtype=D.int64) * (len(array) - 1)

    indices = D.zeros_like(val, dtype=D.int64)
    msk1 = val <= D.gather(array, jlower)
    msk2 = val >= D.gather(array, jupper)
    indices[msk1] = jlower[msk1]
    indices[msk2] = jupper[msk2]

    not_conv = (jupper - jlower) > 1

    while D.any(not_conv):
        jmid = (jupper + jlower) // 2
        mid_vals = D.gather(array, jmid)
        msk1 = val > mid_vals
        msk2 = val <= mid_vals
        jlower[msk1] = jmid[msk1]
        jupper[msk2] = jmid[msk2]
        not_conv = (jupper - jlower) > 1

    return jlower


class BlockTimer():
    """Timing Class

    Takes advantage of the with syntax in order to time a block of code.
    
    Parameters
    ----------
    section_label : str
        name given to section of code
    start_now : bool
        if True the timer is started upon construction, otherwise start() must be called.
    suppress_print : bool
        if True a message will be printed upon destruction with the section_label and the code time.
    """

    def __init__(self, section_label=None, start_now=True, suppress_print=False):
        self.start_now = start_now
        self.stopped = not start_now
        self.start_time = None
        self.end_time = None
        self.label = section_label
        self.suppress_print = suppress_print

    def __enter__(self):
        if self.start_now:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if not self.stopped:
            self.end()
        if not self.suppress_print:
            if self.start_time is None and not self.start_now:
                print("Timer was never started, cannot give run-time of code.")
                print("Set start_now=True to count from start of block to end")
            elif self.start_now:
                if self.label: print("Section:\n\t" + self.label)
                self.end_time = time.perf_counter()
                print("\tThis code block took {}".format(convert_suffix(self.end_time - self.start_time)))
            else:
                if self.label: print("Section:\n\t" + self.label)
                print("\tBetween start() and end(), the time taken was {}".format(
                    convert_suffix(self.end_time - self.start_time)))

    def start(self):
        """Method to start the timer
        """
        self.start_time = time.perf_counter()

    def end(self):
        """Method to stop the timer
        """
        self.end_time = time.perf_counter()
        self.stopped = True

    def elapsed(self):
        """Method to get the elapsed time.
        
        Returns
        -------
        float
            Returns the elapsed time since timer start if stop() was not called, otherwise returns the elapsed time
            between timer start and when elapsed() is called.        
        """

        if self.end_time is None:
            return time.perf_counter() - self.start_time
        else:
            return self.end_time - self.start_time

    def restart_timer(self):
        """Method to restart the timer. 
        
        Sets the start time to now and resets the end time.
        """
        if self.stopped:
            self.stopped = False
        self.start_time = time.perf_counter()
        self.end_time = None
