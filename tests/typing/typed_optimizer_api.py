"""Typed algorithm fixtures for ``tests/typing/test_typing.py``.

These are plain modules (not ``test_*``) so pytest does not run them; the test
invokes mypy on this file and asserts it reports no errors.
"""
import clypto as cy


class NewStyleSearch(cy.DecoratedOptimizer):
    """The typed way to write a new-style optimizer: inherit the base."""

    scale = cy.Argument(float, (0.0, 1.0), 0.5)

    def initialize(self) -> None:
        shape = (len(self.population), self.bounds.ndim)
        width = self.bounds.ub - self.bounds.lb
        self.population.solutions = self.bounds.lb + width * self.rng.random(shape)

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
    """The classic base already types ``self.pop``, ``self.problem``, etc."""

    def __init__(self, epoch: int = 100, pop_size: int = 30, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])

    def evolve(self, epoch: int) -> None:
        self.g_best = self.get_best_agent(self.pop, self.problem.minmax)


@cy.legacy
class DecoratedClassic(ClassicSearch):
    """The legacy decorator on an explicit base keeps the base members typed."""

    def evolve(self, epoch: int) -> None:
        _ = len(self.pop)
