# Tutorial: the decorator optimizer API

This step-by-step tutorial builds a complete optimizer from scratch with
`@cy.optimizer`, gives its agents extra state with `@cy.agent`, runs it, and
compiles it with Cython. It complements the reference-style [Tutorial](tutorial.md)
and the [migration guide](migration.md).

## 1. The problem

Every optimizer solves a `Problem`: an objective function, a search space, and a
direction. We minimize the sphere function in 30 dimensions.

```python
import numpy as np
import clypto as cy


def sphere(solution):
    return np.sum(solution ** 2)


problem = cy.Problem(
    obj_func=sphere,
    bounds=cy.FloatVar(lb=[-10.0] * 30, ub=[10.0] * 30),
    minmax="min",
)
```

## 2. A first optimizer

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

## 3. Arguments are validated

`scale` accepts only floats in the half-open range `(0.0, 1.0)`, because a
`tuple` bound is exclusive and a `list` bound is inclusive — the same convention
as the legacy validator. Passing a bad value fails immediately, and unknown
arguments are rejected:

```python
RandomSearch(scale=2.0)     # TypeError: out of range
RandomSearch(beta=1)        # TypeError: unexpected argument
```

Read back the resolved configuration from `optimizer.parameters`, and the search
space from `optimizer.bounds` (`lb`, `ub`, `ndim`).

## 4. Working with the population

`self.population` is a `cy.Population`. The most useful operations:

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

## 5. Custom agent state

When an algorithm needs per-agent state (velocity, memory, a tag, ...), declare
an agent with `cy.Attribute` and pass it to the optimizer:

```python
@cy.agent
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle)
class MyPSO:
    inertia: cy.Argument[float, (0.0, 1.5), 0.7]

    def generate_agent(self, solution=None):
        agent = super().generate_agent(solution)
        agent.velocity = self.rng.normal(0, 1, self.bounds.ndim)
        return agent

    def evolve(self, epoch):
        for agent in self.population:
            agent.velocity = (self.inertia * agent.velocity
                              + self.rng.normal(0, 1, self.bounds.ndim))
            agent.solution = agent.solution + agent.velocity
```

Overriding `generate_agent` is optional; without it, each new agent gets the
declared defaults.

## 6. Reuse classic algorithms with `@cy.legacy`

The classic MEALPY-style API is still available. Decorate a class instead of
naming a base, and `super().__init__`, `self.validator`, `self.pop` and
`generate_empty_agent` work exactly as before:

```python
@cy.legacy
class ClassicRandomSearch:
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(self.problem.generate_solution(encoded=True))
            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)
            self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
```

## 7. Compile it

Install the `compile` extra once, then opt in per class with `compile=True` (or
`precompile=True` on `@cy.legacy`):

```bash
pip install "clypto[compile]"
```

```python
@cy.agent(compile=True)
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle, compile=True)
class MyPSO:
    ...
```

The class is built into a native extension on import and cached by source hash.
Without Cython or a compiler the flags raise (`ImportError`/`RuntimeError`)
instead of silently running uncompiled.

## Next steps

- Convert existing code with the [migration guide](migration.md).
- See the full API surface in the reference [Tutorial](tutorial.md).
- Use `seed=` on every `solve` call for reproducible runs.
