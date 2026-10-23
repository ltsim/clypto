# Migrating algorithms

This guide covers the 2026 optimizer-API changes and how to move code onto them.
Nothing about the built-in algorithms' mathematics changed: every catalog
optimizer returns the same result for the same seed.

## 1. The classic API (`cimport clypto.core as cy`)

A classic algorithm is a Cython extension type on `cy.Optimizer`; `evolve` is
the C method `cdef void evolve(self, int epoch)`:

```cython
cimport clypto.core as cy


cdef class RandomSearch(cy.Optimizer):
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])

    cdef void evolve(self, int epoch):
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for agent in self.population.toarray():
            x = cy.correct_solution(self.problem, self.problem.generate_solution())
            n_population.append(self.population.create_agent(x))
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
```

Moving an algorithm of the previous classic API onto it:

| Before | After |
| --- | --- |
| `def evolve(self, epoch)` (Python subclass of `cy.LegacyOptimizer`) | `cdef void evolve(self, int epoch)` in a `.pyx`; `solve()` rejects a Python `evolve` |
| `agent.copy()` | `cy.duplicate_agent(agent)` (the agent's static `cdef clone`) |
| `agent.update(field=value, ...)` | `agent.field = value` (typed `cdef public` fields) |
| `XAgent(solution, field=value)` (`**fields`) | an explicit `__init__(self, solution=None, ..., field=None)` on the agent class |
| `self.population.correct_solution(x)` | `cy.correct_solution(self.problem, x)` |
| `cy.ResetPopulation` / an `amend_solution` override | `cy.reset_solution(self.problem, self.generator, x)` / a module-level repair function |
| `self.population.opposite_solution(a, g)` | `cy.opposite_solution(self.problem, self.generator, a, g)` |
| `self.population.duplicate()` / `pop_new = []` | `cy.snapshot(self.population)` / `cy.empty_snapshot(self.population)` |
| `for idx in range(0, pop_size): ... self.population[idx]` | `for idx, agent in enumerate(self.population.toarray()): ... agent` |
| `self.population = [agents]` | `self.population = self.population.spawn([agents])` (a typed C field) |
| `self.mode not in self.AVAILABLE_MODES` | `self.mode == "sequential"` |
| `mode=None` / `"thread"` / `"process"` | `mode="sequential"` (default) / `"parallel"` |
| a Population subclass with Python attributes | typed `cdef public` fields and a `cdef void copy_state(self, cy.Population new)` override |
| `cy.get_all_optimizers(engine=...)`, `clypto.native.collection.{legacy,vectorize}.*` | `cy.get_all_optimizers()`, `clypto.native.collection.<category>.*` |

## 2. From the MEALPY-style API

`@cy.legacy`, `self.validator` and the `Target` class are gone; the hooks are
public again and the helpers moved to the population, the agent or `cy`:

| Before | After |
| --- | --- |
| `@cy.legacy` / `@cy.legacy(precompile=True)` | a `.pyx` with `cimport clypto.core as cy` (section 1) |
| `super().__init__(**kwargs)` + `self._set_parameters([...])` + `self.sort_flag = X` | `super().__init__(parameters=[...], sort_flag=X, **kwargs)` |
| `self.validator.check_int("n", v, bound)` (and `check_float/str/bool`) | `cy.validator(int, v, bound, "n")` |
| `self.pop_size = self.validator.check_int("pop_size", ...)` | `self.population = cy.population(pop_size, range=[5, 10000])` |
| `self.pop_size` | `self.population.size()` (`model.pop_size` still reads it) |
| `_evolve`, `_initialize_variables`, `_initialization`, `_before_main_loop` | `evolve`, `initialize_variables`, `initialization`, `before_main_loop` |
| `self._generate_empty_agent(x)` / `self._generate_agent(x)` / `self._generate_population(n)` | `self.population.create_agent(x)` / `.generate_agent(x)` / `.generate(n)` |
| overriding `_generate_empty_agent` / `_amend_solution` | a `cy.Population` subclass with `create_agent`, passed as `cls=` / `cy.reset_solution` or a repair function |
| `self._correct_solution(x)` | `cy.correct_solution(self.problem, x)` |
| `agent.target = self._get_target(agent.solution)` | `agent.evaluate(self.problem)` |
| `t = self._get_target(x)` | `candidate = self.population.evaluate_solution(x)` (an evaluated `Agent`) |
| `agent.update(solution=x, target=t)` | `agent.update_solution(candidate)` |
| `agent.target.fitness` / `.objectives` | `agent.fitness` / `.objectives` (read-only) |
| `self._compare_target(t1, t2, sense)` | `cy.is_better(a1, a2, sense)` |
| `self._get_better_agent(a, b, sense)` | `cy.get_better_agent(a, b, sense)` |
| `self._get_sorted_population(pop, sense)` | `self.population.sort()` / `cy.sort_agents(agents, sense)` |
| `self._greedy_selection_population(old, new, sense)` | `self.population.greedy(new, self.mode)` / `cy.greedy_agents(old, new, sense, self.mode)` |
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
