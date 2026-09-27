"""Helpers for algorithms: hyper-parameter validation and the random operators.

``cy.validator(int, epoch, [1, 100000], "epoch")``: a ``tuple`` bound is
exclusive, a ``list`` inclusive. The operators take the optimizer's
``generator`` so they draw from the seeded stream.
"""
import math
import numbers
from operator import itemgetter

import numpy as np

cdef double EPSILON = 10e-10


cdef bint _in_bound(object value, object bound):
    if isinstance(bound, tuple):
        exclusive = True
    elif isinstance(bound, list):
        exclusive = False
    else:
        raise TypeError(f"'bound' must be a tuple (exclusive) or list (inclusive), got {type(bound).__name__}.")
    if len(bound) != 2:
        raise ValueError(f"'bound' must have exactly 2 elements, got {len(bound)}.")
    low, high = bound
    if exclusive:
        return (low == -math.inf or low < value) and (high == math.inf or value < high)
    return (low == -math.inf or low <= value) and (high == math.inf or value <= high)


cdef bint _is_real_number(object value):
    return isinstance(value, numbers.Number) and not isinstance(value, bool)


cpdef object validator(object dtype, object value, object bound=None, str name="value"):
    """``value`` as ``dtype`` (``int``, ``float``, ``str`` or ``bool``) when it lies in ``bound``, else ``TypeError``.

    Numbers: ``bound`` is a ``tuple`` (exclusive) or ``list`` (inclusive) pair.
    ``str``: ``bound`` lists the allowed values. ``bool``: default ``(True, False)``.
    """
    if dtype is int or dtype is float:
        if _is_real_number(value) and (bound is None or _in_bound(value, bound)):
            return dtype(value)
        suffix = "" if bound is None else f" and value should be in range: {bound}"
        raise TypeError(f"'{name}' should be {'an integer' if dtype is int else 'a float'}{suffix}.")
    if dtype is str:
        if isinstance(value, str) and (bound is None or value in bound):
            return value
        suffix = "" if bound is None else f" and value should be one of: {bound}"
        raise TypeError(f"'{name}' should be a string{suffix}.")
    if dtype is bool:
        bound = (True, False) if bound is None else bound
        # `is` comparisons avoid 1/0 being treated as True/False.
        if isinstance(value, bool):
            for allowed in bound:
                if value is allowed:
                    return value
        raise TypeError(f"'{name}' should be a boolean, one of: {bound}.")
    raise TypeError(f"cy.validator does not support {dtype!r}.")


cpdef object check_is_int_and_float(str name, object value, object bound_int=None, object bound_float=None):
    """An ``int`` in ``bound_int`` or a ``float`` in ``bound_float`` (e.g. a count or a ratio)."""
    # Order matters: bool is a subclass of int, so reject it up front.
    if isinstance(value, bool):
        pass
    elif isinstance(value, (int, np.integer)):
        if bound_int is None or _in_bound(value, bound_int):
            return int(value)
    elif isinstance(value, (float, np.floating)):
        if bound_float is None or _in_bound(value, bound_float):
            return float(value)
    int_suffix = "" if bound_int is None else f" in range: {bound_int}"
    float_suffix = "" if bound_float is None else f" in range: {bound_float}"
    raise TypeError(f"'{name}' can be int{int_suffix}, or float{float_suffix}.")


cpdef object levy_flight(object generator, double beta=1.0, double multiplier=0.001, object size=None, int case=0):
    """Levy-flight step of shape ``size``.

    ``beta`` in [0, 2] (0-1 exploit, 1-2 explore). ``case`` 0 scales by a
    uniform draw, 1 by a normal draw, -1 not at all.
    """
    sigma_u = np.power(
        math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2)
        / (math.gamma((1 + beta) / 2.0) * beta * np.power(2.0, (beta - 1) / 2)),
        1.0 / beta,
    )
    size = 1 if size is None else size
    u = generator.normal(0, sigma_u, size)
    v = generator.normal(0, 1, size)
    s = u / np.power(np.abs(v) + EPSILON, 1 / beta)
    if case == 0:
        step = multiplier * s * generator.uniform()
    elif case == 1:
        step = multiplier * s * generator.normal(0, 1)
    else:
        step = multiplier * s
    return step[0] if size == 1 else step


cpdef int roulette_wheel(object generator, str sense, object fitness):
    """Roulette-wheel index for a min or max problem; handles negative fitness."""
    fitness = np.array(fitness).ravel()
    if np.ptp(fitness) == 0:
        return int(generator.integers(0, len(fitness)))
    if np.any(fitness < 0):
        fitness = fitness - np.min(fitness)
    final_fitness = fitness
    if sense == "min":
        final_fitness = np.max(fitness) - fitness
    prob = final_fitness / np.sum(final_fitness)
    return int(generator.choice(range(0, len(fitness)), p=prob))


cpdef list kway_tournament(object generator, str sense, object agents, object k_way=0.2, int output=2, bint reverse=False):
    """Indexes of the ``output`` best (``reverse``: worst) of ``k_way`` random agents."""
    if 0 < k_way < 1:
        k_way = int(k_way * len(agents))
    list_id = generator.choice(range(len(agents)), int(k_way), replace=False)
    list_parents = sorted(
        [[idx, agents[idx].fitness] for idx in list_id],
        key=itemgetter(1), reverse=sense != "min",
    )
    if reverse:
        return [parent[0] for parent in list_parents[-output:]]
    return [parent[0] for parent in list_parents[:output]]


cpdef list split_groups(object agents, Py_ssize_t n_groups, Py_ssize_t m_agents):
    """``n_groups`` groups of ``m_agents`` consecutive agents (copies)."""
    return [[agent.copy() for agent in agents[idx * m_agents:(idx + 1) * m_agents]] for idx in range(n_groups)]
