"""Cython-only helpers cimported by the native collection as ``cy``."""
from clypto.optimizer.validator import Validator


cdef object validator(object dtype, object value, object bound=None, str name="value"):
    """Validate a hyper-parameter: ``cy.validator(int, epoch, [1, 100000], "epoch")``.

    ``bound`` follows the Validator convention: a ``tuple`` is exclusive, a
    ``list`` inclusive.
    """
    if dtype is int:
        return Validator.check_int(name, value, bound)
    if dtype is float:
        return Validator.check_float(name, value, bound)
    if dtype is bool:
        return Validator.check_bool(name, value, bound or (True, False))
    if dtype is str:
        return Validator.check_str(name, value, bound)
    raise TypeError(f"cy.validator does not support {dtype!r}.")
