#!/usr/bin/env python
"""Tests for cy.problem decorator, factory, and decode=True support."""

import numpy as np
import pytest

import clypto as cy
from clypto.native.collection.swarm_based import PSO

pytest.importorskip("Cython")


def test_problem_function_decorator_with_bounds():
    @cy.problem(
        bounds=cy.NumberBounds(float, low=[-5.0] * 3, up=[5.0] * 3),
        sense="min",
        compile=True,
    )
    def sphere(x):
        return float(np.sum(x**2))

    assert isinstance(sphere, cy.Problem)
    assert sphere.bounds.n_dims == 3
    assert sphere.sense == "min"

    sol = np.array([1.0, 2.0, 3.0])
    res = sphere(sol)
    assert res == pytest.approx(14.0)
    assert sphere.n_evals == 1

    model = PSO.OriginalPSO(epoch=10, pop_size=20)
    best = model.solve(sphere, seed=42)
    assert best.fitness < 1.0


def test_problem_function_deferred_bounds():
    @cy.problem(sense="min", compile=True)
    def rastrigin(x):
        A = 10.0
        return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

    bounds = cy.NumberBounds(float, low=[-5.12] * 2, up=[5.12] * 2)
    p = rastrigin(bounds=bounds)

    assert isinstance(p, cy.Problem)
    assert p.bounds.n_dims == 2
    assert p(np.zeros(2)) == pytest.approx(0.0)


def test_problem_class_decorator_with_evaluate():
    @cy.problem(compile=True)
    class QuadraticShift:
        bounds = cy.NumberBounds(float, low=[-10.0] * 4, up=[10.0] * 4)
        sense = "min"

        def __init__(self, shift=2.0):
            self.shift = shift

        def evaluate(self, solution):
            return float(np.sum((solution - self.shift) ** 2))

    prob = QuadraticShift(shift=3.0)
    assert isinstance(prob, cy.Problem)
    assert prob.bounds.n_dims == 4
    assert prob(np.full(4, 3.0)) == pytest.approx(0.0)

    model = PSO.OriginalPSO(epoch=15, pop_size=30)
    best = model.solve(prob, seed=42)
    assert best.fitness < 1.0


CITIES = ["A", "B", "C", "D"]
DIST_MATRIX = np.array([
    [0.0, 2.0, 9.0, 10.0],
    [1.0, 0.0, 6.0, 4.0],
    [15.0, 7.0, 0.0, 8.0],
    [6.0, 3.0, 12.0, 0.0],
])


def test_problem_decode_permutation_tsp():
    cities = CITIES
    dist_matrix = DIST_MATRIX

    @cy.problem(
        bounds=cy.PermutationBounds(cities, name="tour"),
        sense="min",
        decode=True,
        compile=True,
    )
    def tsp(decoded):
        tour = decoded["tour"]
        idx = [cities.index(c) for c in tour]
        cost = sum(dist_matrix[idx[i], idx[i + 1]] for i in range(len(idx) - 1))
        return float(cost + dist_matrix[idx[-1], idx[0]])

    assert tsp.decode is True

    # Test single evaluation
    keys = np.array([0.5, 0.1, 0.9, 0.2])
    score = tsp(keys)
    assert isinstance(score, float)
    assert tsp.n_evals == 1

    # Solve with optimizer
    model = PSO.OriginalPSO(epoch=10, pop_size=20)
    best = model.solve(tsp, seed=42)
    decoded_best = tsp.decode_solution(best.solution)
    assert len(decoded_best["tour"]) == 4


def test_problem_decode_mixed_engineering_case_d():
    @cy.problem(decode=True, compile=True)
    class PressureVesselDesign:
        bounds = cy.Bounds(
            cy.NumberBounds(float, low=[10.0, 10.0], up=[50.0, 50.0], name="dimensions"),
            cy.NumberBounds(int, low=1, up=5, name="thickness"),
            cy.StringBounds(("carbon", "titanium"), name="material"),
        )
        sense = "min"

        def evaluate(self, solution):
            dims = solution["dimensions"]
            thickness = solution["thickness"]
            mat = solution["material"]

            base_cost = 1.0 if mat == "carbon" else 3.0
            volume = dims[0] * dims[1] * thickness
            return float(volume * base_cost)

    prob = PressureVesselDesign()
    assert prob.decode is True
    assert prob.bounds.n_dims == 4

    # Test evaluation
    sample = prob.generate_solution(encoded=True)
    cost = prob(sample)
    assert isinstance(cost, float)
    assert cost > 0

    model = PSO.OriginalPSO(epoch=10, pop_size=20)
    best = model.solve(prob, seed=42)
    assert best.fitness > 0


def test_problem_vectorized_support():
    @cy.problem(
        bounds=cy.NumberBounds(float, low=[-5.0] * 3, up=[5.0] * 3),
        sense="min",
        vectorize=True,
        compile=True,
    )
    def sphere_vec(X):
        return np.sum(X**2, axis=1)

    assert sphere_vec.vectorized is True

    X = np.ones((5, 3))
    fitness, O = sphere_vec.evaluate(X)
    assert fitness.shape == (5,)
    assert np.allclose(fitness, [3.0] * 5)
