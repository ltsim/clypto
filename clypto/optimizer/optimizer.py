import abc
import typing

import numpy as np
import numpy.typing as npt

from clypto.agents.virtual import BaseAgent
from clypto.utils.history import TrackHistory
from clypto.utils.problem import Problem
from clypto.utils.target import Target
from clypto.utils.termination import Termination


class Optimizer(abc.ABC):
    """
    Abstract base class that defines the contract for metaheuristic optimization
    algorithms.

    This class establishes the standard lifecycle of an optimizer, from variable
    setup and population initialization to the main evolutionary loop, termination
    checking, and history tracking. Concrete subclasses (e.g., genetic algorithms,
    particle swarm, differential evolution) must implement every abstract method
    to provide their specific search strategy.

    The expected execution flow is:

        1. ``check_problem``           — validate and bind the target problem.
        2. ``initialize_variables``    — set up algorithm-specific state.
        3. ``before_initialization``   — pre-population hook (e.g., seeding).
        4. ``initialization``          — create the initial population.
        5. ``after_initialization``    — post-population hook (e.g., evaluation).
        6. ``before_main_loop``        — pre-loop hook (e.g., reset counters).
        7. ``evolve(epoch)``           — run one generation; repeat until
                                         ``check_termination`` signals stop.
        8. ``track_optimize_process``  — finalize and persist history.

    Notes:
        - Subclasses should override ``amend_solution`` to apply algorithm-specific
          repair strategies. ``correct_solution`` should NOT be overridden; it
          enforces problem-level constraints on top of ``amend_solution``.
        - The ``seed`` parameter in ``solve`` must be set explicitly to an int
          for reproducible runs.
    """

    @property
    @abc.abstractmethod
    def history(self) -> TrackHistory:
        ...

    @abc.abstractmethod
    def initialize_variables(self) -> None:
        """
        Initialize internal state and algorithm-specific variables.

        Called once per ``solve`` invocation, before any population is created.
        Typical use: allocate counters, reset best-so-far trackers, and configure
        algorithm hyperparameters that depend on the bound problem (e.g., problem
        dimensionality).
        """
        ...

    @abc.abstractmethod
    def before_initialization(
            self,
            starting_solutions: typing.Sequence[float] | npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Hook executed immediately before population initialization.

        Args:
            starting_solutions: Optional user-supplied starting positions. May be
                a 1D sequence (single solution) or a 2D array with shape
                ``(pop_size, n_dims)``. Providing this is **not recommended**:
                it can bias the search and break the algorithm's diversity
                assumptions. Leave as ``None`` for standard random initialization.
        """
        ...

    @abc.abstractmethod
    def initialization(self) -> None:
        """
        Create the initial population of agents.

        Populates ``self.pop`` (or its equivalent) with ``pop_size`` agents,
        either randomly within the problem bounds or from the starting
        solutions passed to ``before_initialization``.
        """
        ...

    @abc.abstractmethod
    def after_initialization(self) -> None:
        """
        Hook executed immediately after the initial population is built.

        Typical use: evaluate the initial population's targets, identify the
        first global best, and prepare any data structures the main loop will
        consume on its first iteration.
        """
        ...

    @abc.abstractmethod
    def before_main_loop(self) -> None:
        """
        Hook executed once just before the evolutionary loop begins.

        Useful for resetting per-run counters, starting timers, or logging the
        starting state of the optimizer.
        """
        ...

    @abc.abstractmethod
    def evolve(self, epoch: int) -> None:
        """
        Execute one generation (epoch) of the optimization process.

        This is the core of each algorithm: subclasses implement their
        search operators here (selection, mutation, crossover, velocity
        updates, etc.), evaluate the resulting agents, and update the
        population and best-so-far records.

        Args:
            epoch: The 1-based index of the current generation.
        """
        ...

    @abc.abstractmethod
    def check_problem(self, problem: dict | Problem, seed: int | None):
        """
        Validate the problem definition and bind it to the optimizer.

        Accepts either a ``Problem`` instance or a dictionary describing the
        objective function, bounds, dimensionality, and optimization direction
        (minimize/maximize). Also seeds the random number generator when
        ``seed`` is provided.

        Args:
            problem: Problem instance or dictionary with the problem definition.
            seed: Optional integer seed for reproducibility.
        """
        ...

    @abc.abstractmethod
    def check_termination(self, mode="start", termination=None, epoch=None):
        """
        Configure and evaluate termination conditions.

        Args:
            mode: ``"start"`` to initialize/validate the termination criteria
                at the beginning of a run, or ``"end"`` (or similar) to test
                whether stopping conditions have been met at the end of an
                epoch.
            termination: A ``Termination`` instance or dictionary describing
                the stopping criteria (max epochs, max function evaluations,
                target fitness, time budget, etc.).
            epoch: The current epoch, used when evaluating epoch-based
                termination conditions.
        """
        ...

    @abc.abstractmethod
    def solve(
            self,
            problem: dict | Problem,
            termination: typing.Optional[dict | Termination] = None,
            starting_solutions: typing.Sequence[float] | npt.NDArray[np.float64] | None = None,
            seed: int | None = None,
            debug: bool = False,
    ) -> "BaseAgent":
        """
        Run the full optimization process end to end.

        Orchestrates the lifecycle described in the class docstring: it
        validates the problem, initializes the population, runs the
        evolutionary loop until termination is met, and returns the best
        agent found.

        Args:
            problem: A ``Problem`` instance or a dictionary describing the
                objective and search space.
            termination: A ``Termination`` instance or dictionary specifying
                the stopping criteria. If ``None``, the optimizer's default
                criteria are used.
            starting_solutions: Optional list or 2D numpy array of starting
                positions with length equal to the ``pop_size`` parameter.
                Use with care — it can bias the search.
            seed: Integer seed for the random number generator. Must be set
                **explicitly** to an int for reproducible results.
            debug: If ``True``, the optimizer records detailed
                history of each epoch (best fitness, diversity metrics,
                runtime, etc.) for later inspection.

        Returns:
            The best agent found during the run. The solution and its
            objective value are available via ``g_best.solution`` and
            ``g_best.target`` respectively.
        """
        ...

    @abc.abstractmethod
    def track_optimize_step(
            self,
            pop: list["BaseAgent"] | None = None,
            epoch: int | None = None,
            runtime: float | None = None,
    ) -> None:
        """
        Record per-epoch history and optionally print training details.

        Called once at the end of each evolutionary epoch when
        ``track_optimize=True`` was passed to ``solve``.

        Args:
            pop: The current population at the end of the epoch.
            epoch: The 1-based index of the epoch just completed.
            runtime: Wall-clock runtime for the epoch, in seconds.
        """
        ...

    @abc.abstractmethod
    def track_optimize_process(self) -> None:
        """
        Finalize history tracking after the optimization process finishes.

        Aggregates per-epoch records into the final history structure
        (e.g., convergence curves, runtime totals, list of best agents
        per epoch) for analysis or plotting.
        """
        ...

    @abc.abstractmethod
    def generate_empty_agent(self, solution: np.ndarray | None = None) -> "BaseAgent":
        """
        Create a new agent skeleton without evaluating its target.

        Useful when you need a placeholder agent (e.g., as the destination
        for an operator's output) before the objective function is called.

        Args:
            solution: Optional initial solution vector. If ``None``, a random
                solution within the problem bounds is generated.

        Returns:
            A new agent whose ``solution`` is set but whose ``target`` has
            not yet been computed.
        """
        ...

    @abc.abstractmethod
    def generate_agent(self, solution: np.ndarray | None = None) -> "BaseAgent":
        """
        Create a fully evaluated agent.

        Builds the agent via ``generate_empty_agent`` and then evaluates the
        objective function on its solution, incrementing the function
        evaluation counter.

        Args:
            solution: Optional initial solution vector. If ``None``, a random
                solution within the problem bounds is generated.

        Returns:
            A new agent with both ``solution`` and ``target`` populated.
        """
        ...

    @abc.abstractmethod
    def generate_population(self, pop_size: int | None = None) -> list["BaseAgent"]:
        """
        Generate a population of fully evaluated agents.

        Args:
            pop_size: Number of agents to create. If ``None``, the optimizer's
                configured ``pop_size`` is used.

        Returns:
            A list of agents, each with its ``solution`` and ``target``
            evaluated.
        """
        ...

    @abc.abstractmethod
    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Apply the optimizer's strategy for handling invalid solutions.

        Subclasses override this method to implement their preferred repair
        scheme (e.g., clipping to bounds, reflection, random re-sampling,
        wrapping). It does **not** apply problem-specific constraints —
        those are handled by ``correct_solution``.

        Args:
            solution: A candidate solution that may violate the search bounds.

        Returns:
            A solution adjusted according to the optimizer's strategy.
        """
        ...

    @abc.abstractmethod
    def correct_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Produce a fully valid solution ready for objective evaluation.

        Combines ``amend_solution`` (optimizer-level repair) with any
        problem-specific constraints (e.g., integer rounding, discrete
        mappings, equality/inequality constraints).

        **Do not override** this method in subclasses — override
        ``amend_solution`` instead.

        Args:
            solution: A raw candidate solution.

        Returns:
            A corrected solution that satisfies both bounds and problem
            constraints, safe to pass to the objective function.
        """
        ...

    @abc.abstractmethod
    def update_target_for_population(self, pop: list["BaseAgent"]) -> list["BaseAgent"]:
        """
        Re-evaluate the objective value for every agent in a population.

        Useful after operators that mutate solutions in place, or in dynamic
        problems where the objective changes over time.

        Args:
            pop: The population of agents to update.

        Returns:
            The same population with each agent's ``target`` refreshed.
        """
        ...

    @abc.abstractmethod
    def get_target(self, solution: np.ndarray, counted: bool = True) -> "Target":
        """
        Evaluate the objective function for a single solution.

        Args:
            solution: The real-valued, already-corrected solution vector.
            counted: If ``True``, the function evaluation counter is
                incremented. Set to ``False`` for diagnostic evaluations
                that should not count against the budget.

        Returns:
            The ``Target`` value, encapsulating the objective(s) and any
            associated fitness information.
        """
        ...
