"""Typed algorithm fixtures for ``tests/typing/test_typing.py``.

These are plain modules (not ``test_*``) so pytest does not run them; the test
invokes mypy on this file and asserts it reports no errors.
"""
import clypto as cy


class NewStyleSearch(cy.DecoratedOptimizer):
    """The typed way to write a new-style optimizer: inherit the base."""

    scale = cy.Argument(float, (0.0, 1.0), 0.5)

    def initialize(self) -> None:
        shape = (len(self.population), self.bounds.n_dims)
        width = self.bounds.up - self.bounds.low
        self.population.solutions = self.bounds.low + width * self.rng.random(shape)

    def evolve(self, epoch: int) -> None:
        for idx in range(len(self.population)):
            candidate = self.generate_agent()
            self.population[idx].solution = candidate.solution


@cy.optimizer
class DecoratedSearch(cy.DecoratedOptimizer):
    """The decorator may still be applied; with an explicit base it is a no-op."""

    def evolve(self, epoch: int) -> None:
        _ = len(self.population)


class ClassicSearch(cy.LegacyOptimizer):
    """The classic base already types ``self.population``, ``self.problem``, etc.

    A classic algorithm implements ``cdef void evolve`` in Cython; this Python class only checks the types.
    """

    def __init__(self, epoch: int = 100, pop_size: int = 30, **kwargs: object) -> None:
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])

    def before_main_loop(self) -> None:
        # Python may override the lifecycle hooks; evolve itself is a C method (cdef void evolve)
        self.g_best = self.population.sort()[0]
