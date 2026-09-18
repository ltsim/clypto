# Your first optimizer

Every optimizer solves a `Problem`: an objective function, a search space, and a
direction. We minimize the sphere function in 30 dimensions.

```python
--8<-- "docs/custom-optimizers/_snippets/problem.py"
```

## A first optimizer

Decorate a plain class with `@cy.optimizer` and implement `evolve`. Hyper-parameters
are declared as class annotations with `cy.Argument[type, bound, default]`; the
base class supplies `epoch` and `pop_size`, the population, the RNG, and `solve`.

The bracket syntax accepts `[type]`, `[type, bound]` and
`[type, bound, default]`. Use `...` to skip the bound while keeping a default,
e.g. `cy.Attribute[float, ..., 0.0]`.

```python
@cy.optimizer
class RandomSearch:
    scale: cy.Argument[float, (0.0, 1.0), 0.5]

    def evolve(self, epoch):
        for idx in range(len(self.population)):
            candidate = self.generate_agent()
            if candidate.fitness < self.population[idx].fitness:
                self.population[idx].solution = candidate.solution
```

Run it in one line:

```python
optimizer = RandomSearch(epoch=200, pop_size=50)
g_best = optimizer.solve(problem, seed=7)

print(g_best.solution)        # best vector found
print(g_best.fitness)         # its objective value
```

`solve` returns the best agent, also available as `optimizer.g_best` and
`optimizer.population.best`.

Inside the optimizer the base class exposes:

| Attribute | Meaning |
| --- | --- |
| `self.population` | The `Population` container (see [The population](03-population.md)) |
| `self.rng` | A seeded `numpy.random.Generator` (`seed=` from `solve`) |
| `self.problem` | The bound `Problem` |
| `self.bounds` | `lb`/`ub`/`ndim` view of the search space |
| `self.g_best` | Best agent after `solve` returns |

### Static typing (mypy)

`@cy.optimizer` injects the base class at runtime, which a type checker cannot
see: on a bare decorated class, `self.population`/`self.generate_agent` are
unknown. When you type-check your algorithms, name the base explicitly — the
decorator becomes optional, and `DecoratedOptimizer` provides the same API:

```python
class RandomSearch(cy.DecoratedOptimizer):
    scale: cy.Argument[float, (0.0, 1.0), 0.5]

    def evolve(self, epoch: int) -> None:
        for idx in range(len(self.population)):
            ...
```

For the classic API the base is `cy.LegacyOptimizer` (this is how the catalog
algorithms are written), which makes `self.pop`, `self.g_best` and
`self.problem` statically known as well.

---

Next: [Hyper-parameters](02-parameters.md).
