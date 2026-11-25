"""Regression: ``n_evals`` must equal the objective work in every path.

The lazy ``n_objs`` probe evaluates one throwaway solution; it used to be the one
objective call the counter missed, so ``nf_counter`` under-reported real work by
one on every fresh ``Problem`` (and the ``max_fe`` budget overshot by a whole
population on the exact-fit case).
"""

import numpy as np

import clypto as cy

N_DIMS = 5
POP_SIZE = 5
EPOCH = 3


class Counter:
    """Objective that counts the solutions it evaluates (a vectorized call carries a batch)."""

    def __init__(self):
        self.calls = 0
        self.rows = 0

    def __call__(self, x):
        a = np.asarray(x, dtype=float)
        self.calls += 1
        self.rows += 1 if a.ndim == 1 else a.shape[0]
        return float(np.sum(a**2)) if a.ndim == 1 else np.sum(a**2, axis=-1)


def bounds():
    return cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS)


def model():
    return cy.get_all_optimizers()["OriginalPSO"](pop_size=POP_SIZE, epoch=EPOCH)


def assert_counts(counter, evaluated):
    """``evaluated`` solutions were run through the objective; the counter must agree."""
    assert counter.rows == evaluated, f"real objective work {counter.rows} != problem.n_evals {evaluated}"


def test_fresh_scalar_problem_counts_the_probe():
    counter = Counter()
    problem = cy.Problem(obj_func=counter, bounds=bounds(), sense="min", seed=1)
    model().solve(problem, seed=1)
    assert_counts(counter, problem.n_evals)


def test_second_solve_of_the_same_problem_counts():
    counter = Counter()
    problem = cy.Problem(obj_func=counter, bounds=bounds(), sense="min", seed=1)
    model().solve(problem, seed=1)
    counter.rows = counter.calls = 0  # second run on the same (already probed) Problem
    model().solve(problem, seed=1)
    assert_counts(counter, problem.n_evals)


def test_dict_problem_counts_the_probe():
    counter = Counter()
    problem = {"obj_func": counter, "bounds": bounds(), "sense": "min", "seed": 1}
    solver = model()
    solver.solve(problem, seed=1)
    assert_counts(counter, solver.problem.n_evals)


def test_vectorized_problem_counts_every_row():
    counter = Counter()
    problem = cy.Problem(obj_func=counter, bounds=bounds(), sense="min", seed=1, vectorized=True)
    model().solve(problem, seed=1)
    assert_counts(counter, problem.n_evals)  # n_evals counts solutions, not obj_func invocations
