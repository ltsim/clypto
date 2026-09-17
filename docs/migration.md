# Migrating algorithms

This guide covers the 2026 optimizer-API change and how to move code onto it.
Nothing about the built-in algorithms' mathematics changed: the catalog is
untouched apart from the name of the class it inherits from.

## 1. `Optimizer` is now `LegacyOptimizer`

The classic base class was renamed so that `Optimizer` can stay available as a
backward-compatible alias while the decorator API claims the `optimizer` name.

```python
# Before
from clypto.optimizer import Optimizer


class MyAlgorithm(Optimizer):
    ...


# After
from clypto.optimizer import LegacyOptimizer


class MyAlgorithm(LegacyOptimizer):
    ...
```

`cy.Optimizer` is an alias of `cy.LegacyOptimizer`, so existing imports and
`issubclass` checks keep working:

```python
import clypto as cy

assert cy.Optimizer is cy.LegacyOptimizer
```

Every class in `clypto.collection.*` was migrated with this rename only — no
algorithm body, variable, or hyper-parameter was changed. If you subclass a
catalog optimizer, you do not need to change anything.

## 2. Keep classic code with `@cy.legacy`

You no longer have to name the base class at all. Decorate with `@cy.legacy`
and the decorator injects `LegacyOptimizer` for you, while `super().__init__`,
`self.validator`, `self.pop`, `correct_solution`, `generate_empty_agent` and the
rest of the classic surface keep working:

```python
import clypto as cy


@cy.legacy
class MyAlgorithm:
    def __init__(self, epoch=100, pop_size=30, **kwargs):
        super().__init__(**kwargs)          # reaches LegacyOptimizer
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

The old `@cy.precompile` decorator is replaced by the compilation flags on the
decorators: use `@cy.legacy(precompile=True)` to compile a classic class.

## 3. Port a classic algorithm to `@cy.optimizer`

The decorator API removes the constructor, the validator boilerplate, and the
legacy `self.pop` list. The mapping is mechanical:

| Classic API | Decorator API |
| --- | --- |
| `class X(Optimizer)` | `@cy.optimizer` on a plain class |
| `__init__` + `self.validator.check_int/...` | `name: cy.Argument[type, bound, default]` |
| `self.set_parameters([...])` | automatic (`optimizer.parameters`) |
| `self.pop` (list) | `self.population` (`Population`) |
| `self.generator` (NumPy) / `self.rng` (random) | `self.rng` (`numpy.random.Generator`) |
| `initialization()` / `before_main_loop()` | `initialize()` |
| `agent.target = self.get_target(pos)` | `agent.solution = pos` (evaluates automatically) |
| `agent.solution` / `agent.target.fitness` | `agent.solution` / `agent.fitness` |
| `self.generate_empty_agent(...)` | `self.generate_agent(...)` |
| `self.get_better_agent(a, b, minmax)` | compare `a.fitness` / `b.fitness` |
| `correct_solution(...)` | `problem.correct_solution(...)` |

Before:

```python
from clypto.optimizer import LegacyOptimizer


class RandomSearch(LegacyOptimizer):
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

### `initialize` replaces the constructor

There is no `__init__` in the decorator API — set up state in `initialize`,
which runs after the population exists:

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

If `initialize` is omitted, the population is generated uniformly at random
inside the problem bounds.

### Agent attributes

If the algorithm needs per-agent state, declare an agent and pass it to the
optimizer. Use `generate_agent` to seed attributes:

```python
@cy.agent
class Particle:
    velocity: cy.Attribute[float, ..., 0.0]


@cy.optimizer(agent=Particle)
class MyPSO:
    def generate_agent(self, solution=None):
        agent = super().generate_agent(solution)
        agent.velocity = self.rng.normal(0, 1, self.bounds.ndim)
        return agent

    def evolve(self, epoch):
        for agent in self.population:
            agent.velocity += self.rng.normal(0, 1, self.bounds.ndim)
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


@cy.legacy(precompile=True)
class MyClassic:
    ...
```

Install the `compile` extra (`pip install "clypto[compile]"`). Without Cython or
a compiler, plain (uncompiled) classes still run; the compile flags fail loudly
instead of silently skipping the build.

## 5. Checklist

- [ ] Replace `Optimizer` with `LegacyOptimizer`, or drop the base and add `@cy.legacy`.
- [ ] Replace `@cy.precompile` with `@cy.legacy(precompile=True)` (or the decorator API's `compile=True`).
- [ ] For new algorithms, prefer `@cy.optimizer` + `cy.Argument`; remember there is no `__init__`.
- [ ] Use `self.population`, `self.rng`, `self.bounds`, and `agent.fitness` instead of `self.pop`, `self.generator`, and `agent.target.fitness`.
- [ ] Never assign `agent.fitness`; assign `agent.solution` and it is recomputed.
- [ ] Run the tests with `uv sync --extra dev --extra compile && uv run pytest tests/`.
