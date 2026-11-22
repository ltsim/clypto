"""Regression: ``solve()`` must not discard ``Problem(seed=...)``.

``Problem.coerce`` used to assign ``problem.seed = seed`` unconditionally, so
``model.solve(problem)`` (no seed argument) wiped the Problem's seed, reseeded the
bounds sampler with ``None`` and turned a seeded run into a silently
unreproducible one. The Problem's seed is now the run's default seed.
"""

import numpy as np

import clypto as cy

N_DIMS = 5


def sphere(solution):
    """Module-level objective: sum of squares."""
    return float(np.sum(np.asarray(solution) ** 2))


def run(pass_seed):
    """One run of a fresh ``Problem(seed=1)``; returns (post-solve seed, fitness, solution)."""
    problem = cy.Problem(
        obj_func=sphere,
        bounds=cy.NumberBounds(float, low=[-5.0] * N_DIMS, up=[5.0] * N_DIMS),
        sense="min",
        seed=1,
    )
    model = cy.get_all_optimizers()["OriginalPSO"](pop_size=10, epoch=20)
    g_best = model.solve(problem, seed=1) if pass_seed else model.solve(problem)
    return problem.seed, float(g_best.fitness), np.asarray(g_best.solution).copy()


def test_solve_without_seed_keeps_the_problem_seed():
    """Two identical ``Problem(seed=1)`` + ``solve(problem)`` runs must match bit for bit."""
    seed_a, fit_a, sol_a = run(pass_seed=False)
    seed_b, fit_b, sol_b = run(pass_seed=False)

    assert seed_a == 1, f"Problem.coerce discarded the Problem's seed: problem.seed={seed_a!r}"
    assert fit_a == fit_b, f"two identical seeded runs differ: {fit_a!r} != {fit_b!r}"
    assert np.array_equal(sol_a, sol_b), f"solutions differ:\n{sol_a}\n!=\n{sol_b}"


def test_solve_without_seed_matches_solve_with_the_problem_seed():
    """``solve(problem)`` and ``solve(problem, seed=1)`` must be the same run."""
    _, fit_default, sol_default = run(pass_seed=False)
    _, fit_explicit, sol_explicit = run(pass_seed=True)

    assert fit_default == fit_explicit, f"{fit_default!r} != {fit_explicit!r}"
    assert np.array_equal(sol_default, sol_explicit)
