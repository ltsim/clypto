# Hyper-parameters

Continuing from [Your first optimizer](01-first-optimizer.md), `RandomSearch`
declares one hyper-parameter, `scale`, as a `cy.Argument`. This part covers how
that declaration is validated.

```python
--8<-- "docs/custom-optimizers/_snippets/problem.py"


@cy.optimizer
class RandomSearch:
    scale: cy.Argument[float, (0.0, 1.0), 0.5]

    def evolve(self, epoch):
        ...


optimizer = RandomSearch(epoch=200, pop_size=50)
```

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

---

Next: [The population](03-population.md).
