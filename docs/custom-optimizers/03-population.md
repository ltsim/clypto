# The population

The optimizer's solutions live in `self.population`, a `cy.Population`. The same
`problem` from the previous parts is used here.

```python
--8<-- "docs/custom-optimizers/_snippets/problem.py"
```

## The container

The most useful operations on `self.population`:

```python
self.population.solutions        # (n_pop, ndim) matrix; assigning re-evaluates all
self.population.fitness          # (n_pop,) vector
self.population.best             # best agent (minmax-aware)
self.population.worst            # worst agent
self.population.remove(agent.id) # drop one agent
self.population.append(agent)    # add an agent
self.population.generate()       # create a fresh, evaluated agent (not appended)
```

The typical search loop removes the worst agent and replaces it:

```python
def evolve(self, epoch):
    worst = self.population.worst
    self.population.remove(worst.id)
    self.population.append(self.population.generate())
```

## Seeding the population

You can also set the whole matrix at once, which is convenient in `initialize`:

```python
@cy.optimizer
class Seeded:
    def initialize(self):
        self.population.solutions = self.rng.uniform(
            self.bounds.lb, self.bounds.ub,
            (len(self.population), self.bounds.ndim),
        )

    def evolve(self, epoch):
        ...
```

If `initialize` is omitted, the population is already initialized uniformly at
random inside the bounds.

### Fitness is derived, never assigned

`agent.fitness` is read-only. The only way to change it is to assign
`agent.solution`, which re-evaluates the objective — including in-place maths:

```python
self.population[idx].solution /= 2.0   # setter runs; fitness stays correct
```

---

Next: [Custom agent state](04-agent-state.md).
