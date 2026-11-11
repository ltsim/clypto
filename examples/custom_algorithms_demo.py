#!/usr/bin/env python
"""Demonstration of custom algorithms in Clypto:

1. Pure Python with `@cy.optimizer`
2. Pure Python subclassing `cy.Optimizer`
3. Cython JIT compilation via `clypto.compile()`
"""

import numpy as np
import clypto as cy


def sphere(solution: np.ndarray) -> float:
    """Sphere benchmark objective function."""
    return float(np.sum(solution**2))


# Define problem
problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.NumberBounds(float, low=[-5.0] * 5, up=[5.0] * 5),
    sense="min",
)


# ---------------------------------------------------------------------------
# 1. Pure Python: @cy.optimizer decorator
# ---------------------------------------------------------------------------
@cy.optimizer
class GaussianMutationSearch:
    """Custom pure-Python optimizer with parameter validation."""

    step_size: cy.Argument[float, (0.0, 2.0), 0.2]

    def evolve(self, epoch: int):
        for agent in self.population:
            noise = self.rng.normal(0, self.step_size, size=self.bounds.n_dims)
            candidate_solution = self.bounds.correct(agent.solution + noise)
            candidate = self.generate_agent(candidate_solution)
            if candidate.fitness < agent.fitness:
                agent.solution = candidate.solution


# ---------------------------------------------------------------------------
# 2. Pure Python: Subclassing cy.Optimizer
# ---------------------------------------------------------------------------
class StepDecaySearch(cy.Optimizer):
    """Custom pure-Python optimizer subclassing cy.Optimizer."""

    def __init__(self, epoch: int = 40, pop_size: int = 20, decay: float = 0.9, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.decay = decay

    def evolve(self, epoch: int):
        for agent in self.population:
            agent.solution = self.bounds.correct(agent.solution * self.decay)
            agent.evaluate(self.problem)


# ---------------------------------------------------------------------------
# 3. Cython JIT: clypto.compile()
# ---------------------------------------------------------------------------
CYTHON_SOURCE = """# cython: language_level=3
cimport clypto.core as cy
import numpy as np

cdef class NativeContractionSearch(cy.Optimizer):
    cdef public double factor

    def __init__(self, epoch=40, pop_size=20, factor=0.8, **kwargs):
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        self.factor = cy.validator(float, factor, (0.0, 1.0), "factor")

    cdef void evolve(self, int epoch):
        cdef cy.Agent agent
        for agent in self.population:
            agent.solution = agent.solution * self.factor
            agent.evaluate(self.problem)
"""


def main():
    print("=== 1. Running GaussianMutationSearch (@cy.optimizer) ===")
    opt1 = GaussianMutationSearch(epoch=40, pop_size=20, step_size=0.15)
    best1 = opt1.solve(problem, seed=42)
    print(f"Best fitness: {best1.fitness:.6e}")
    print(f"Best vector:  {best1.solution[:3]}...")

    print("\n=== 2. Running StepDecaySearch (subclassing cy.Optimizer) ===")
    opt2 = StepDecaySearch(epoch=40, pop_size=20, decay=0.85)
    best2 = opt2.solve(problem, seed=42)
    print(f"Best fitness: {best2.fitness:.6e}")
    print(f"Best vector:  {best2.solution[:3]}...")

    print("\n=== 3. JIT-Compiling and Running NativeContractionSearch ===")
    NativeContractionSearch = cy.compile(CYTHON_SOURCE)
    opt3 = NativeContractionSearch(epoch=40, pop_size=20, factor=0.75)
    best3 = opt3.solve(problem, seed=42)
    print(f"Best fitness: {best3.fitness:.6e}")
    print(f"Best vector:  {best3.solution[:3]}...")


if __name__ == "__main__":
    main()
