#!/usr/bin/env python
"""Decorator-path fixture for the memory-leak probe (memory audit, 2026-11).

Lives in benchmarks/ because ``@cy.optimizer(compile=True)`` needs a real,
importable source file (``inspect.getsource``); the JIT digest is stable so
repeated probe iterations hit the compile cache.
"""
import clypto as cy


@cy.agent(compile=True)
class DecAgent:
    drift: cy.Attribute[float, (0.0, 1.0), 0.5]  # type: ignore[valid-type, type-arg, syntax]


@cy.optimizer(agent=DecAgent, compile=True)
class DecaySearch:
    step_size: cy.Argument[float, (0.0, 2.0), 0.2]  # type: ignore[valid-type, type-arg, syntax]

    def evolve(self, epoch: int):
        for agent in self.population:  # type: ignore[attr-defined]
            noise = self.rng.normal(0, self.step_size, size=self.bounds.n_dims)  # type: ignore[attr-defined]
            candidate = self.generate_agent(  # type: ignore[attr-defined]
                self.bounds.correct(agent.solution + noise)  # type: ignore[attr-defined]
            )
            if candidate.fitness < agent.fitness:
                agent.solution = candidate.solution
