# Classic API and compilation

## Reuse classic algorithms with `@cy.legacy`

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

## Compile it

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
- See the full API surface in the reference [Tutorial](../tutorial.md).
- Use `seed=` on every `solve` call for reproducible runs.
