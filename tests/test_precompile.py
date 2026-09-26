import importlib.machinery
import sys

import numpy as np
import pytest

import clypto as cy
from clypto.optimizer.precompile import precompile

pytest.importorskip("Cython")

N_DIMS = 3


def objective(solution):
    return np.sum(solution**2)


@precompile
class RandomSearch(cy.LegacyOptimizer):
    def __init__(self, epoch=50, pop_size=25, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self._set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def _evolve(self, epoch):
        for idx in range(self.pop_size):
            pos_new = self._correct_solution(
                self.problem.generate_solution(encoded=True)
            )
            agent = self._generate_empty_agent(pos_new)
            agent.target = self._get_target(pos_new)
            self.pop[idx] = self._get_better_agent(
                self.pop[idx], agent, self.problem.sense
            )


@pytest.fixture(scope="module")
def problem():
    return cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="min",
    )


def test_precompile_returns_compiled_class():
    assert RandomSearch.__clypto_precompiled__ is True
    assert issubclass(RandomSearch, cy.LegacyOptimizer)


def test_precompile_actually_compiled():
    assert RandomSearch.__module__.startswith("_clypto_")

    extension = sys.modules[RandomSearch.__module__]
    assert extension.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))

    assert type(RandomSearch._evolve).__name__ == "cython_function_or_method"
    assert type(RandomSearch.__init__).__name__ == "cython_function_or_method"


def test_precompile_rejects_non_optimizer():
    class NotAnOptimizer:
        pass

    with pytest.raises(TypeError):
        precompile(NotAnOptimizer)


def test_precompile_raises_without_cython(monkeypatch):
    runtime = sys.modules["clypto.optimizer.precompile._runtime"]

    class TempOptimizer(cy.LegacyOptimizer):
        def _evolve(self, epoch):
            pass

    def no_cython():
        raise ImportError("no Cython")

    monkeypatch.setattr(runtime, "cython_version", no_cython)

    with pytest.raises(ImportError):
        precompile(TempOptimizer)


def test_precompiled_optimizer_solves(problem):
    g_best = RandomSearch(epoch=50, pop_size=25).solve(problem, seed=1)

    assert g_best.solution.shape == (N_DIMS,)
    assert np.all(np.isfinite(g_best.solution))
    assert np.isfinite(g_best.target.fitness)


def test_precompiled_optimizer_tracking(problem):
    model = RandomSearch(epoch=10, pop_size=15)
    model.solve(problem, seed=1, debug=True, track_population=True)

    tracker = model.tracker
    assert tracker.group.attrs["n_epochs"] == 10
    assert tracker["global_best_fit"].shape == (10,)
    assert tracker["solution"].shape == (10, 15, N_DIMS)
    assert np.isfinite(tracker["solution"]).all()

