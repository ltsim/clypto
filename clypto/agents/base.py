import abc
import typing

from clypto.hints.array import NDArrayType
from clypto.hints.sense import SenseType
from clypto.utils.target import Target


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def copy(self) -> "BaseAgent": ...

    @abc.abstractmethod
    def update_agent(self, solution: NDArrayType, target: Target) -> "BaseAgent": ...

    @abc.abstractmethod
    def update(self, *args: typing.Any, **kwargs: typing.Any) -> "BaseAgent": ...

    @abc.abstractmethod
    def sync_if_duplicate(self, other: "BaseAgent") -> bool: ...

    @abc.abstractmethod
    def _compare_fitness(self, other: "BaseAgent", minmax: SenseType = "min") -> int:
        """
        Compare fitness between self and other.

        Returns:
            -1 if self is better
             0 if equal
             1 if other is better
        """
        ...

    @abc.abstractmethod
    def get_better_solution(
        self, other: "BaseAgent", minmax: SenseType = "min"
    ) -> "BaseAgent":
        """
        Return better solution

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self if self._compare_fitness(other, minmax) <= 0 else other

    @abc.abstractmethod
    def is_better_than(self, other: "BaseAgent", minmax: SenseType = "min") -> bool:
        """
        Compare the current agent with other agent. Return True if current agent is better and False otherwise

        Args:
            other: The compared agent
            minmax: The problem
        """
        ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __eq__(self, other: typing.Any) -> bool:
        """Check if two agents are equal based on their solutions with a tolerance."""
        ...

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Generate a hash based on the solution of the agent.
        This is useful for using agents in sets or as dictionary keys."""
        ...

    @abc.abstractmethod
    def __float__(self) -> float:
        """Compute the fitness value from self and other."""
        ...
