# Cython-only helpers for the native collection:
#     from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native.target cimport NativeTarget


cdef object validator(object dtype, object value, object bound=*, str name=*)


cdef inline bint compare_target(NativeTarget x, NativeTarget y, str minmax):
    """True when ``x`` is better than (``"max"``: at least as good as) ``y``."""
    if minmax == "min":
        return x.fitness < y.fitness
    return not (x.fitness < y.fitness)
