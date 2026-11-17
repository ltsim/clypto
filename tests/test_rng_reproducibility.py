#!/usr/bin/env python
"""Seeded runs must replay: no sampler and no lazy probe may draw outside the stream.

Two regressions from the RNG/NFE audit:

- ``OriginalPSS`` built its Latin-Hypercube sample with ``scipy.stats.qmc`` and no
  seed, so ``qmc`` drew from its own uncontrolled generator and two ``seed=1`` runs
  diverged. It now hands the optimizer's generator to the sampler.
- ``Problem.n_objs`` probes with one random solution on first access, and
  ``Agent.evaluate`` reads ``obj_weights`` right after each evaluation, so the probe
  used to fire inside initialisation and consume one draw: the initial population
  was draws ``[1, 3, 4, 5, 6]`` instead of ``[1, 2, 3, 4, 5]``.
"""

import numpy as np

import clypto as cy


def sphere(solution):
    """Module-level objective: sum of squares."""
    return float(np.sum(np.asarray(solution) ** 2))


def test_pss_lhs_same_seed_is_reproducible():
    """Two fresh ``OriginalPSS`` runs with the same seed must match bit for bit."""
    pss = cy.get_all_optimizers()["OriginalPSS"]  # the flat API does not export it

    def run():
        problem = cy.Problem(
            obj_func=sphere,
            bounds=cy.NumberBounds(float, low=[-5.0] * 5, up=[5.0] * 5),
            sense="min",
            seed=1,
        )
        model = pss(pop_size=20, epoch=20)  # LHS is the default sampling_method
        g_best = model.solve(problem, seed=1)
        return g_best.fitness, np.asarray(g_best.solution).copy(), model.problem.n_evals

    fit_a, sol_a, nevals_a = run()
    fit_b, sol_b, nevals_b = run()

    assert fit_a == fit_b, f"fitness differs across identical seed=1 runs: {fit_a!r} != {fit_b!r}"
    assert np.array_equal(sol_a, sol_b), f"solutions differ:\n{sol_a}\n!=\n{sol_b}"
    assert nevals_a == nevals_b, f"n_evals differ: {nevals_a} != {nevals_b}"


def test_n_objs_probe_does_not_consume_bounds_draws():
    """The probe evaluates one solution but must leave the bounds stream untouched."""
    recorded = []

    def objective(solution):
        # Record every argument the objective sees, the throwaway probe included.
        recorded.append(float(np.asarray(solution).ravel()[0]))
        return float(np.sum(np.asarray(solution) ** 2))

    # A 1-dim [0, 1] draw is bit-identical to ``default_rng(seed).random(k)``.
    problem = {
        "obj_func": objective,
        "bounds": cy.NumberBounds(float, low=[0.0], up=[1.0]),
        "sense": "min",
    }
    model = cy.get_all_optimizers()["OriginalPSO"](pop_size=5, epoch=1)
    model.solve(problem, seed=1)

    expected = np.random.default_rng(1).random(5)

    # Call 0 is agent 0, call 1 is the probe, calls 2..5 are agents 1..4.
    assert len(recorded) >= 6, f"expected agent 0 + probe + 4 agents, got {recorded}"
    population = [recorded[0], *recorded[2:6]]

    assert np.array_equal(population, expected), (
        f"initial population is not bounds draws 1..5 (the probe consumed a draw):\n  population = {population}\n  expected   = {expected.tolist()}\n  full order = {recorded}"
    )
    # The probe is a real objective call, so it stays uncounted in n_evals for now.
    assert len(recorded) - model.problem.n_evals == 1
