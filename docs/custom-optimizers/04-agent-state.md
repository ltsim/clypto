# Custom agent state

Results so far only carry a `solution`. When an algorithm needs per-agent state
(velocity, memory, a tag, ...), declare an agent with `cy.Attribute` and pass it
to the optimizer.

```python
--8<-- "docs/custom-optimizers/_snippets/problem.py"
```

```python
@cy.agent
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle)
class MyPSO:
    inertia: cy.Argument[float, (0.0, 1.5), 0.7]

    def generate(self, agent, solution):
        agent.velocity = self.rng.normal(0, 1, self.bounds.ndim)
        return agent

    def evolve(self, epoch):
        for agent in self.population:
            agent.velocity = (self.inertia * agent.velocity
                              + self.rng.normal(0, 1, self.bounds.ndim))
            agent.solution = agent.solution + agent.velocity
```

Overriding `generate` is optional; without it, each new agent gets the declared
defaults. The hook receives the freshly created agent — declared defaults already
applied, not yet evaluated — plus the sampled `solution`, and returns the agent
to keep.

---

Next: [Classic API and compilation](05-legacy-and-compilation.md).
