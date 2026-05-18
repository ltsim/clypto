import abc
import typing

import numpy as np
import numpy.typing as npt

from mealprint.utils.termination import Termination
from mealprint.utils.problem import Problem
from mealprint.utils.agent import Agent
from mealprint.utils.target import Target


class Optimizer(abc.ABC):
    """

    """
    @abc.abstractmethod
    def initialize_variables(self) -> None:
        """
        """
        ...

    @abc.abstractmethod
    def before_initialization(self, starting_solutions: typing.Sequence[float] | npt.NDArray[np.float64] | None = None) -> None:
        """
        Args:
            starting_solutions: The starting solutions (not recommended)
        """
        ...

    @abc.abstractmethod
    def initialization(self) -> None:
        """

        """
        ...

    @abc.abstractmethod
    def after_initialization(self) -> None:
        """

        """
        ...

    @abc.abstractmethod
    def before_main_loop(self) -> None:
        """

        """
        ...

    @abc.abstractmethod
    def evolve(self, epoch: int) -> None:
        """

        """
        ...

    @abc.abstractmethod
    def check_problem(self, problem: dict | Problem, seed: int | None):
        """

        """
        ...

    @abc.abstractmethod
    def check_termination(self, mode="start", termination=None, epoch=None):
        """

        """
        ...

    @abc.abstractmethod
    def solve(self, problem: dict | Problem,
              termination: typing.Optional[dict | Termination] = None,
              starting_solutions: typing.Sequence[float] | npt.NDArray[np.float64] | None = None,
              seed: int | None = None, track_optimize: bool = False) -> Agent:
        """
        Args:
            problem: an instance of Problem class or a dictionary
            termination: The termination dictionary or an instance of Termination class
            starting_solutions: List or 2D matrix (numpy array) of starting positions with length equal pop_size parameter
            seed: seed for random number generation needed to be *explicitly* set to int value
            track_optimize: Track optimize in history
        Returns:
            g_best: g_best, the best found agent, that hold the best solution and the best target. Access by: .g_best.solution, .g_best.target
        """
        ...

    @abc.abstractmethod
    def track_optimize_step(self, pop: list[Agent] |  None = None, epoch: int | None = None, runtime: float| None = None) -> None:
        """
        Save some historical data and print out the detailed information of training process in each epoch

        Args:
            pop: the current population
            epoch: current iteration
            runtime: the runtime for current iteration
        """
        ...

    @abc.abstractmethod
    def track_optimize_process(self) -> None:
        """
        Save some historical data after training process finished
        """
        ...

    @abc.abstractmethod
    def generate_empty_agent(self, solution: np.ndarray | None = None) -> Agent:
        """
        Generate new agent with solution

        Args:
            solution (np.ndarray): The solution
        """
        ...

    @abc.abstractmethod
    def generate_agent(self, solution: np.ndarray | None = None) -> Agent:
        """
        Generate new agent with full information

        Args:
            solution (np.ndarray): The solution
        """
        ...

    @abc.abstractmethod
    def generate_population(self, pop_size: int | None = None) -> list[Agent]:
        """
        Args:
            pop_size (int): number of solutions

        Returns:
            list: population or list of solutions/agents
        """
        ...

    @abc.abstractmethod
    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        This function is based on optimizer's strategy.
        In each optimizer, this function can be overridden

        Args:
            solution: The position

        Returns:
            The valid solution based on optimizer's strategy
        """
        ...

    @abc.abstractmethod
    def correct_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        This function is based on optimizer's strategy and problem-specific condition
        DO NOT override this function

        Args:
            solution: The position

        Returns:
            The correct solution that can be used to calculate target
        """
        ...

    @abc.abstractmethod
    def update_target_for_population(self, pop: list[Agent]) -> list[Agent]:
        """
        Update target for the input population

        Args:
            pop: the population of agents

        Returns:
            list: population with updated target value
        """
        ...

    @abc.abstractmethod
    def get_target(self, solution: np.ndarray, counted: bool = True) -> Target:
        """
        Get target value

        Args:
            solution: The real-value solution
            counted: Indicating the number of function evaluations is increasing or not

        Returns:
            The target value
        """
        ...
