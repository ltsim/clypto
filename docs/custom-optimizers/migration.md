# Migrating algorithms

This guide covers the 2026 optimizer-API changes and how to move code onto them.
Nothing about the built-in algorithms' mathematics changed: every catalog
optimizer returns the same result for the same seed.

## 1. The classic API (`cy.LegacyOptimizer`)

A classic algorithm subclasses `cy.LegacyOptimizer` (`cy.Optimizer` when written
in Cython with `cimport clypto.core as cy`):

```python
import clypto as cy


class RandomSearch(cy.LegacyOptimizer):
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])

    def evolve(self, epoch):
        pop_size = self.population.size()
        for idx in range(pop_size):
            pos_new = self.population.correct_solution(self.problem.generate_solution())
            candidate = self.population.evaluate_solution(pos_new)
            if candidate.fitness < self.population[idx].fitness:
                self.population[idx] = candidate
```

## 2. From the MEALPY-style API

`@cy.legacy`, `self.validator` and the `Target` class are gone; the hooks are
public again and the helpers moved to the population, the agent or `cy`:

| Before | After |
| --- | --- |
| `@cy.legacy` / `@cy.legacy(precompile=True)` | subclass `cy.LegacyOptimizer`; compile a `.pyx` with `cimport clypto.core as cy` |
| `super().__init__(**kwargs)` + `self._set_parameters([...])` + `self.sort_flag = X` | `super().__init__(parameters=[...], sort_flag=X, **kwargs)` |
| `self.validator.check_int("n", v, bound)` (and `check_float/str/bool`) | `cy.validator(int, v, bound, "n")` |
| `self.pop_size = self.validator.check_int("pop_size", ...)` | `self.population = cy.population(pop_size, range=[5, 10000])` |
| `self.pop_size` | `self.population.size()` (`model.pop_size` still reads it) |
| `_evolve`, `_initialize_variables`, `_initialization`, `_before_main_loop` | `evolve`, `initialize_variables`, `initialization`, `before_main_loop` |
| `self._generate_empty_agent(x)` / `self._generate_agent(x)` / `self._generate_population(n)` | `self.population.create_agent(x)` / `.generate_agent(x)` / `.generate(n)` |
| overriding `_generate_empty_agent` / `_amend_solution` | a `cy.Population` subclass with `create_agent` / `amend_solution`, passed as `cls=` |
| `self._correct_solution(x)` | `self.population.correct_solution(x)` |
| `agent.target = self._get_target(agent.solution)` | `agent.evaluate(self.problem)` |
| `t = self._get_target(x)` | `candidate = self.population.evaluate_solution(x)` (an evaluated `Agent`) |
| `agent.update(solution=x, target=t)` | `agent.update_solution(candidate)` |
| `agent.target.fitness` / `.objectives` | `agent.fitness` / `.objectives` (read-only) |
| `self._compare_target(t1, t2, sense)` | `cy.is_better(a1, a2, sense)` |
| `self._get_better_agent(a, b, sense)` | `cy.get_better_agent(a, b, sense)` |
| `self._get_sorted_population(pop, sense)` | `self.population.sort()` / `cy.sort_agents(agents, sense)` |
| `self._greedy_selection_population(old, new, sense)` | `self.population.greedy(new)` / `cy.greedy_agents(old, new, sense)` |
| `self._update_target_for_population(agents)` | `self.population.evaluate(agents, self.mode)` |
| `self._get_levy_flight_step(...)` | `cy.levy_flight(self.generator, ...)` |
| `self._get_index_roulette_wheel_selection(f)` | `cy.roulette_wheel(self.generator, self.problem.sense, f)` |
| `self.nf_counter` | unchanged; evaluations are counted by `problem.n_evals` |

## 3. Port a classic algorithm to `@cy.optimizer`

The decorator API removes the constructor and the validator boilerplate. The
mapping is mechanical:

| Classic API | Decorator API |
| --- | --- |
| `class X(cy.LegacyOptimizer)` | `@cy.optimizer` on a plain class |
| `__init__` + `cy.validator(...)` | `name: cy.Argument[type, bound, default]` |
| `super().__init__(parameters=[...])` | automatic (`optimizer.parameters`) |
| `self.generator` (NumPy) / `self.rng` (random) | `self.rng` (`numpy.random.Generator`) |
| `initialization()` / `before_main_loop()` | `initialize()` |
| `evolve(epoch)` | `evolve(epoch)` |
| `agent.evaluate(self.problem)` | `agent.solution = pos` (evaluates automatically) |
| `self.population.create_agent(...)` | `self.generate_agent(...)` |
| `cy.get_better_agent(a, b, sense)` | compare `a.fitness` / `b.fitness` |
| `self.population.correct_solution(...)` | `problem.correct_solution(...)` |

Before: the classic `RandomSearch` of section 1.

After:

```python
import clypto as cy


@cy.optimizer
class RandomSearch:
    def evolve(self, epoch):
        for idx in range(len(self.population)):
            candidate = self.generate_agent()
            if candidate.fitness < self.population[idx].fitness:
                self.population[idx].solution = candidate.solution
```

`epoch` and `pop_size` are provided by the base class, so only algorithm-specific
hyper-parameters need an `Argument`.

> For type-checked code, inherit the base explicitly instead of relying on the
> decorator: `class RandomSearch(cy.DecoratedOptimizer)`. mypy cannot see the
> base that `@cy.optimizer` injects, so only the explicit form types
> `self.population`, `self.generate_agent`, `self.bounds` and `self.rng`.

### `initialize` replaces the constructor

There is no `__init__` in the decorator API — set up state in `initialize`,
which runs after the population exists:

```python
@cy.optimizer
class Seeded:
    def initialize(self):
        self.population.solutions = self.rng.uniform(
            self.bounds.low, self.bounds.up,
            (len(self.population), self.bounds.n_dims),
        )

    def evolve(self, epoch):
        ...
```

If `initialize` is omitted, the population is generated uniformly at random
inside the problem bounds.

### Agent attributes

If the algorithm needs per-agent state, declare an agent and pass it to the
optimizer. Use `generate` to seed attributes:

```python
@cy.agent
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle)
class MyPSO:
    def generate(self, agent, solution):
        agent.velocity = self.rng.normal(0, 1, self.bounds.n_dims)
        return agent

    def evolve(self, epoch):
        for agent in self.population:
            agent.velocity += self.rng.normal(0, 1, self.bounds.n_dims)
            agent.solution = agent.solution + agent.velocity
```

## 4. Compilation

Compilation is opt-in per class:

```python
@cy.agent(compile=True)
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle, compile=True)
class MyPSO:
    ...

```

Install the `compile` extra (`pip install "clypto[compile]"`). Without Cython or
a compiler, plain (uncompiled) classes still run; the compile flags fail loudly
instead of silently skipping the build.
