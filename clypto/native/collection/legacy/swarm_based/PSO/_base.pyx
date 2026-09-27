#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""The particle and the swarm shared by every legacy PSO variant."""
import numpy as np
cimport numpy as cnp
cimport clypto.core as cy


cdef class PSOAgent(cy.Agent):
    """A particle: its position (``solution``/``fitness``), ``velocity`` and personal best (pbest)."""

    def update_velocity(self, cnp.ndarray gbest_solution, w, double c1, double c2, object generator,
                        v_min=None, v_max=None):
        """Inertia + pull towards pbest (cognitive) + pull towards gbest (social), clipped when limits are given."""
        cdef Py_ssize_t dims = self.solution.shape[0]
        cognitive = c1 * generator.random(dims) * (self.pbest_solution - self.solution)
        social = c2 * generator.random(dims) * (gbest_solution - self.solution)
        velocity = w * self.velocity + cognitive + social
        self.velocity = velocity if v_min is None else np.clip(velocity, v_min, v_max)

    def move(self):
        """The position the velocity leads to; the optimizer decides whether the particle goes there."""
        return self.solution + self.velocity

    def update_pbest(self, cy.Agent candidate, str sense):
        """Keep ``candidate`` as the personal best when it beats it (always, the first time)."""
        if self.pbest_solution is not None and not cy.better_fitness(candidate.fitness, self.pbest_fitness, sense):
            return False
        self.pbest_solution = candidate.solution.copy()
        self.pbest_fitness = candidate.fitness
        return True


cdef class PSOPopulation(cy.Population):
    """The swarm: particles start with a random velocity in ``[v_min, v_max]`` = ± half the search range."""

    def bind(self, problem, generator):
        super().bind(problem, generator)
        self.v_max = 0.5 * (problem.bounds.up - problem.bounds.low)
        self.v_min = -self.v_max

    def create_agent(self, solution=None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return PSOAgent(solution, velocity=self.generator.uniform(self.v_min, self.v_max))

    def generate_agent(self, solution=None):
        agent = self.create_agent(solution)
        agent.evaluate(self.problem)
        agent.update_pbest(agent, self.problem.sense)
        return agent


cdef class ResetPSOPopulation(PSOPopulation):
    """A particle that leaves the bounds is sent to a random position (OriginalPSO, LDW_PSO, AIW_PSO)."""

    def amend_solution(self, cnp.ndarray solution):
        low, up = self.problem.bounds.low, self.problem.bounds.up
        return np.where((low <= solution) & (solution <= up), solution, self.generator.uniform(low, up))
