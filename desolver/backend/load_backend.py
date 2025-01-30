import sys
import einops

from desolver.backend.common import *
from desolver.backend.autoray_backend import *
from desolver.backend.numpy_backend import *
try:
    from desolver.backend.torch_backend import *
except ImportError:
    pass

print("Using `autoray` backend", file=sys.stderr)


def contract_first_ndims(a, b, n=1):
    """Contracts the tensors a and b along the first n dimensions

    Simple interface to contract two tensors, a and b, along their first n dimensions.
    For example, for a (2,2,3,9) tensor and a (2,2,7) tensor, an n=1 contraction gives
    a (2,3,9,7) tensor and an n=2 contraction gives a (3,9,7) tensor.

    Parameters
    ----------
    a : array-type
        First tensor to contract with
    b : array-type
        Second tensor to contract with
    n : int
        The first indices to contract along

    Returns
    -------
    array-type
        The resultant tensor

    Raises
    ------
    ValueError
        If `n>len(shape(a))` ie. the contraction requires more dimensions than there exists

    See Also
    --------
    einsum: function used to write tensor operations via einstein notation

    Examples
    --------
    ```python
    >>> a = D.array([[0.0, 1.0],[1.0, 0.0]])
    >>> b = D.array([[2.0, 1.0],[3.0, 5.0]])
    >>> D.contract_first_ndims(a, b, n=1)
    array([3., 1.])
    >>> D.contract_first_ndims(a, b, n=2)
    4.0
    ```
    """
    if len(ar_numpy.shape(a)) > len(ar_numpy.shape(b)):
        a, b = b, a
    if n > len(ar_numpy.shape(a)):
        raise ValueError("Cannot contract along more dims than there exists!")
    na = len(ar_numpy.shape(a))
    nb = len(ar_numpy.shape(b))
    einsum_str = "{},{}->{}"
    estr1 = "".join([chr(97 + i) for i in range(na)])
    estr2 = "".join([chr(97 + i) for i in range(nb)])
    estr3 = "".join([chr(97 + i + n) for i in range(nb - n)])
    einsum_str = einsum_str.format(estr1, estr2, estr3)
    return einops.einsum(a, b, einsum_str)
