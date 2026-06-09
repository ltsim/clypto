import abc
import typing

from clypto.hints.array import NDArrayType
from clypto.utils.target import Target


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def copy(self) -> "BaseAgent": ...

    @abc.abstractmethod
    def update_agent(self, solution: NDArrayType, target: Target) -> "BaseAgent": ...

    @abc.abstractmethod
    def update(self, *args: typing.Any, **kwargs: typing.Any) -> "BaseAgent": ...
