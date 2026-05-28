import abc

import numpy as np
import numpy.typing as npt

from mealpy.utils.target import Target


class BaseAgent(abc.ABC):
    @property
    @abc.abstractmethod
    def target(self) -> Target | None:
        ...

    @target.setter
    @abc.abstractmethod
    def target(self, target: Target | None) -> None:
        ...

    @property
    @abc.abstractmethod
    def solution(self) -> npt.NDArray[np.number] | None:
        ...

    @solution.setter
    @abc.abstractmethod
    def solution(self, solution: npt.NDArray[np.number]):
        ...
