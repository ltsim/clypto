import typing
from clypto.hints.array import NDArrayType

__all__ = ['RuntimeAgent']

class RuntimeAgent:
    """Solution-only agent used by ``@cy.optimizer`` when none is declared.

    This is intentionally a plain Python class (not a Cython ``cdef`` class) so
    that user agent classes created by ``@cy.agent`` can inject it as a base and
    carry arbitrary declared attributes.
    """
    def __init__(self, solution: NDArrayType | None = None, **attributes: typing.Any) -> None: ...
    @property
    def solution(self) -> NDArrayType | None: ...
    @solution.setter
    def solution(self, value: NDArrayType | None) -> None: ...
    @property
    def objectives(self) -> NDArrayType | None: ...
    @property
    def fitness(self) -> float | None: ...
    def copy(self) -> RuntimeAgent: ...
