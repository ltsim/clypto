#!/usr/bin/env python
"""Exercise the nogil/prange parallel update path with a compiled evaluator."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import clypto as cy

pytest.importorskip("Cython")

N_DIMS = 4

_EVALUATOR_PYX = """
from clypto.optimizer.native.nogil cimport _NogilEvaluator


cdef void sphere(const double* x, Py_ssize_t n, double* out) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0.0
    for i in range(n):
        s += x[i] * x[i]
    out[0] = s


cdef class SphereEvaluator(_NogilEvaluator):
    def __cinit__(self):
        self._func = sphere
        self._n_objs = 1
"""


@pytest.fixture(scope="module")
def sphere_evaluator():
    import pyximport

    tmp = Path(tempfile.mkdtemp(prefix="clypto-nogil-"))
    (tmp / "_clypto_nogil_probe.pyx").write_text(_EVALUATOR_PYX)
    pyximport.install(language_level=3)
    sys.path.insert(0, str(tmp))
    try:
        module = __import__("_clypto_nogil_probe")
    finally:
        sys.path.remove(str(tmp))
    return module.SphereEvaluator()


def _problem(evaluator=None):
    return cy.Problem(
        obj_func=lambda s: float(np.sum(s**2)),
        bounds=cy.FloatVar(lb=[-1.0] * N_DIMS, ub=[1.0] * N_DIMS),
        minmax="min",
        evaluator=evaluator,
    )


def test_parallel_mode_runs_with_compiled_evaluator(sphere_evaluator):
    from clypto.collection.swarm_based.ARO import OriginalARO

    model = OriginalARO(epoch=15, pop_size=12, mode="parallel")
    g_best = model.solve(_problem(sphere_evaluator), seed=1)

    assert np.all(np.isfinite(g_best.solution))
    assert np.isfinite(g_best.target.fitness)
    assert g_best.target.fitness == pytest.approx(
        float(np.sum(np.asarray(g_best.solution) ** 2))
    )


def test_parallel_mode_without_evaluator_falls_back():
    from clypto.collection.swarm_based.ARO import OriginalARO

    model = OriginalARO(epoch=15, pop_size=12, mode="parallel")
    g_best = model.solve(_problem(None), seed=1)

    assert np.all(np.isfinite(g_best.solution))


def test_default_mode_is_unchanged():
    from clypto.collection.swarm_based.ARO import OriginalARO

    default = OriginalARO(epoch=15, pop_size=12).solve(_problem(None), seed=1)
    identical = OriginalARO(epoch=15, pop_size=12).solve(_problem(None), seed=1)

    assert default.target.fitness == identical.target.fitness
