"""Regression tests: SHADE-family optimizers must survive a population smaller than 10.

Both classes used to crash on the first epoch for pop_size in [5, 9]:
OriginalSHADE sampled ``uniform(2 / pop_size, 0.2)`` (inverted interval), and
OriginalLSHADEcnEpSin let ``NP`` grow past ``len(population)`` when
``pop_size_min`` exceeded ``pop_size``.
"""

import numpy as np
import pytest

import clypto as cy

N_DIMS = 5
EPOCH = 5
SEED = 1


def objective(solution):
    return np.sum(solution**2)


def run(name, pop_size):
    problem = cy.Problem(
        obj_func=objective,
        bounds=cy.NumberBounds(float, low=[-1.0] * N_DIMS, up=[1.0] * N_DIMS),
        sense="min",
    )
    model = cy.get_all_optimizers()[name](epoch=EPOCH, pop_size=pop_size)
    return problem, model.solve(problem, seed=SEED)


def assert_usable(problem, g_best):
    solution = np.asarray(g_best.solution)
    assert solution.shape == (N_DIMS,)
    assert np.all(np.isfinite(solution))
    assert np.all(problem.bounds.low <= solution) and np.all(solution <= problem.bounds.up)
    assert np.isfinite(float(g_best.fitness))


@pytest.mark.parametrize("pop_size", [5, 9])
def test_original_shade_runs_with_small_population(pop_size):
    problem, g_best = run("OriginalSHADE", pop_size)
    assert_usable(problem, g_best)


@pytest.mark.parametrize("pop_size", [5, 9])
def test_original_lshade_cn_ep_sin_runs_with_small_population(pop_size):
    problem, g_best = run("OriginalLSHADEcnEpSin", pop_size)
    assert_usable(problem, g_best)


@pytest.mark.parametrize("pop_size", [5, 9])
def test_l_shade_still_runs_with_small_population(pop_size):
    # L_SHADE never sampled an inverted interval; pinned so a future edit keeps it that way.
    problem, g_best = run("L_SHADE", pop_size)
    assert_usable(problem, g_best)
