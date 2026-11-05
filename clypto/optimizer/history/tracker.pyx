#!/usr/bin/env python
# Created for clypto's in-memory population tracking.
# --------------------------------------------------%
"""``Tracker``: the per-run history held in ``optimizer.tracker``.

Three classes share the module:

+ :class:`MetricBuffer` — the per-epoch metric values in one ``(epoch, metrics)``
  ``float64`` matrix, allocated for ``optimizer.epoch`` rows when the run starts
  and trimmed to the epochs actually recorded. ``exploration``/``exploitation``
  are derived from ``diversity`` once, when the run closes.
+ :class:`SnapshotBuffer` — the optional population frames (positions, fitness
  and objectives), one ``(pop, ...)`` frame per epoch, grown by doubling and
  NaN-padded on assembly. Frames are sized from the widest epoch, so optimizers
  that grow and shrink their population (SAP_DE, ...) are recorded without
  guessing a capacity in advance.
+ :class:`Tracker` — the facade the engine talks to. It reads ``agent.solution``
  and ``agent.fitness``, decides whether a population is captured, and resolves
  ``tracker["global_best_fit"]`` / ``tracker["solution"]`` by name.

Everything lives in RAM and describes the most recent ``solve()`` run. Run
metadata lives on the tracker's typed fields (``tracker.n_epochs``,
``tracker.optimizer_name``, ...) — ``tracker.meta`` is gone, and no dictionary is
involved: buffers are typed numpy arrays and names resolve to fixed indices.

Per-epoch callbacks are not part of the tracker — pass them to
``solve(before_iteration=..., after_iteration=...)`` instead.
"""
import typing

import numpy as np
cimport numpy as cnp

from clypto.optimizer.native.population cimport Population

cnp.import_array()

METRIC_NAMES = (
    "global_best_fit",
    "current_best_fit",
    "current_worst_fit",
    "mean_fit",
    "std_fit",
    "diversity",
    "exploration",
    "exploitation",
    "epoch_time",
    "nfe",
    "epoch_pop_size",
)
N_METRICS = len(METRIC_NAMES)


class MetricBuffer:
    """Per-epoch metrics in one fixed ``(epoch, metrics)`` matrix.

    ``allocate(epoch)`` only happens once per run (``start``); ``record`` fills a
    row from the population arrays; ``finalize`` derives exploration/exploitation
    and trims the buffer to ``(n_epochs, metrics)`` copies, allocating fresh
    arrays so the buffer can be reused for the next run.
    """

    def __init__(self) -> None:
        self.__buffer: np.ndarray | None = None
        self.__n_epochs = 0
        self.__capacity = 0

    def allocate(self, max_epochs: int) -> None:
        """One ``(max_epochs, n_metrics)`` matrix, ready for ``record``."""
        self.__buffer = np.empty((max(1, int(max_epochs)), N_METRICS), dtype=np.float64)
        self.__capacity = max(1, int(max_epochs))

    def reset(self) -> None:
        """Drop the matrix; ``allocate`` will recreate it on the next start."""
        self.__buffer = None
        self.__n_epochs = 0
        self.__capacity = 0

    @property
    def n_epochs(self) -> int:
        """Number of recorded epochs."""
        return self.__n_epochs

    @property
    def buffer(self) -> np.ndarray | None:
        """The live matrix (or ``None`` before ``allocate``)."""
        return self.__buffer

    def record(self, positions: np.ndarray, fits: np.ndarray,
               runtime: float, nfe: int, global_best_fit: float,
               minimize: bool = True) -> None:
        """Append one epoch's statistics computed from the population arrays."""
        n = positions.shape[0]
        # per-epoch aggregates (C loops in numpy over the ``(pop,)`` arrays)
        c_best, c_worst = (float(np.min(fits)), float(np.max(fits))) if minimize \
            else (float(np.max(fits)), float(np.min(fits)))
        median = np.median(positions, axis=0)
        diversity = float(np.mean(np.abs(median - positions)))

        row = self.__buffer[self.__n_epochs]
        row[METRIC_NAMES.index("global_best_fit")] = float(global_best_fit)
        row[METRIC_NAMES.index("current_best_fit")] = c_best
        row[METRIC_NAMES.index("current_worst_fit")] = c_worst
        row[METRIC_NAMES.index("mean_fit")] = float(np.mean(fits))
        row[METRIC_NAMES.index("std_fit")] = float(np.std(fits))
        row[METRIC_NAMES.index("diversity")] = diversity
        row[METRIC_NAMES.index("epoch_time")] = float(runtime)
        row[METRIC_NAMES.index("nfe")] = int(nfe)
        row[METRIC_NAMES.index("epoch_pop_size")] = int(n)
        self.__n_epochs += 1

    def finalize(self) -> np.ndarray:
        """Derive exploration/exploitation and trim to ``n_epochs`` fresh columns."""
        diversity = self.__buffer[: self.__n_epochs, METRIC_NAMES.index("diversity")].copy()
        if diversity.size and np.max(diversity) > 0:
            exploration = 100.0 * diversity / np.max(diversity)
        else:
            exploration = np.zeros_like(diversity)
        data = self.__buffer[: self.__n_epochs].copy()
        data[:, METRIC_NAMES.index("exploration")] = exploration
        data[:, METRIC_NAMES.index("exploitation")] = 100.0 - exploration
        return data


class SnapshotBuffer:
    """Population frames per epoch, one NaN-padded array per capture field.

    Solutions, fitness and objectives (multi-objective only) are grown by
    doubling when an epoch needs more rows or a wider population than before;
    ``finalize`` copies the exact ``(epochs, max_pop, ...)`` arrays.
    """

    def __init__(self) -> None:
        self.__solution = None
        self.__fitness = None
        self.__objective = None
        self.__n_epochs = 0
        self.__cap_epochs = 0
        self.__cap_pop = 0
        self.__max_pop = 0
        self.__n_objs = None

    def reset(self) -> None:
        self.__solution = None
        self.__fitness = None
        self.__objective = None
        self.__n_epochs = 0
        self.__cap_epochs = 0
        self.__cap_pop = 0
        self.__max_pop = 0
        self.__n_objs = None

    @property
    def n_objs(self) -> int | None:
        """Objectives per agent, decided by the first recorded frame (``None`` before it)."""
        return self.__n_objs

    def record(self, positions: np.ndarray, fits: np.ndarray,
               objectives: np.ndarray | None = None) -> None:
        """Append one epoch's frames (positions, fitness, objectives when multi-objective)."""
        n_pop, n_dims = positions.shape[0], positions.shape[1]
        first = self.__solution is None
        if first:
            self.__n_objs = None if objectives is None else int(objectives.shape[1])
        elif self.__n_objs is None:
            self.__n_objs = None if objectives is None else int(objectives.shape[1])

        n_epochs = self.__n_epochs
        cap_epochs, cap_pop = self.__cap_epochs, self.__cap_pop
        if n_epochs + 1 > cap_epochs or n_pop > cap_pop:
            new_epochs = max(8, n_epochs + 1, cap_epochs * 2)
            new_pop = max(n_pop, cap_pop * 2, 8)
            solution = np.full((new_epochs, new_pop, n_dims), np.nan, dtype=np.float64)
            fitness = np.full((new_epochs, new_pop), np.nan, dtype=np.float64)
            objectives_buf = (
                None if self.__objective is None
                else np.full((new_epochs, new_pop, self.__objective.shape[2]), np.nan, dtype=np.float64)
            )
            if self.__solution is not None:
                solution[:cap_epochs, :cap_pop] = self.__solution
                fitness[:cap_epochs, :cap_pop] = self.__fitness
            if self.__objective is not None:
                objectives_buf[:cap_epochs, :cap_pop] = self.__objective
            self.__solution, self.__fitness, self.__objective = solution, fitness, objectives_buf
            self.__cap_epochs, self.__cap_pop = new_epochs, new_pop
        cap_pop = self.__cap_pop
        self.__solution[n_epochs, :n_pop] = positions
        self.__fitness[n_epochs, :n_pop] = fits
        if objectives is not None and self.__n_objs and self.__n_objs > 1:
            if self.__objective is None:
                self.__objective = np.full(
                    (self.__cap_epochs, cap_pop, int(self.__n_objs)), np.nan, dtype=np.float64
                )
            self.__objective[n_epochs, :n_pop] = objectives
        self.__n_epochs = n_epochs + 1
        self.__max_pop = max(self.__max_pop, n_pop)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """``(solution, fitness, objectives)`` arrays with exact shapes, plus the objective frame."""
        n_epochs, n_pop = self.__n_epochs, self.__max_pop
        solution = self.__solution[:n_epochs, :n_pop].copy()
        fitness = self.__fitness[:n_epochs, :n_pop].copy()
        objectives = (
            self.__objective[:n_epochs, :n_pop].copy()
            if self.__objective is not None and self.__n_objs and self.__n_objs > 1 else None
        )
        self.__solution = self.__fitness = self.__objective = None
        self.__n_epochs = 0
        self.__cap_epochs, self.__cap_pop, self.__max_pop = 0, 0, 0
        return solution, fitness, objectives


class Tracker:
    """The run history of the engine: per-epoch metrics and optional population frames.

    Notes
    ~~~~~
    + Metrics are ``(epoch,)`` arrays read by name (``tracker["global_best_fit"]``);
      population snapshots are ``(epoch, pop, ...)`` arrays, NaN-padded to the widest
      epoch, when ``finalize`` runs.
    + Everything lives in RAM and belongs to the most recent ``solve()``; an
      untracked ``solve`` leaves the tracker in its initial state (``recording``
      false, no arrays, default run metadata).
    + Run metadata lives on the types (``tracker.optimizer_name``,
      ``tracker.problem_name``, ``tracker.seed``, ``tracker.pop_size``,
      ``tracker.n_dims``, ``tracker.sense``, ``tracker.capture_population``,
      ``tracker.n_epochs``, ``tracker.n_objs``) — no dictionary.

    Per-iteration callbacks are not part of the tracker — pass them to
    ``solve(before_iteration=..., after_iteration=...)`` instead.
    """

    def __init__(self) -> None:
        self.optimizer_name: str = ""
        self.problem_name: str = ""
        self.seed: typing.Optional[int] = None
        self.pop_size: int = 0
        self.n_dims: int = 0
        self.sense: str = "min"
        self.capture_population: bool = False
        self.n_epochs: int = 0
        self.n_objs: int = -1
        self.__recording = False
        self.__minimize = True
        self.__metrics = MetricBuffer()
        self.__snapshots = SnapshotBuffer()
        self.__metric_matrix: np.ndarray | None = None
        self.__solution: np.ndarray | None = None
        self.__fitness: np.ndarray | None = None
        self.__objective: np.ndarray | None = None

    def __repr__(self) -> str:
        state = "recording" if self.__recording else ("tracked" if self.n_epochs else "empty")
        return f"{type(self).__name__}({state}, optimizer={self.optimizer_name!r}, n_epochs={self.n_epochs})"

    def reset(self) -> None:
        """Discard everything from the previous run and return to the initial state."""
        self.optimizer_name, self.problem_name, self.seed = "", "", None
        self.pop_size, self.n_dims, self.sense = 0, 0, "min"
        self.capture_population, self.n_epochs, self.n_objs = False, 0, -1
        self.__recording, self.__minimize = False, True
        self.__metrics.reset()
        self.__snapshots.reset()
        self.__metric_matrix = None
        self.__solution = self.__fitness = self.__objective = None

    @property
    def recording(self) -> bool:
        return self.__recording

    def start(self, optimizer, problem, seed=None, capture_population: bool = False) -> None:
        """Begin a new recording; ``meta`` became typed fields on the tracker."""
        self.reset()
        self.__minimize = str(problem.sense) == "min"
        self.capture_population = bool(capture_population)
        self.optimizer_name = str(optimizer.name)
        self.problem_name = str(problem.name)
        self.seed = seed
        self.pop_size = int(optimizer.pop_size)
        self.n_dims = int(problem.n_dims)
        self.sense = str(problem.sense)
        self.__metrics.allocate(optimizer.epoch)
        self.__recording = True

    def record(self, pop, runtime: float, nfe: int, global_best_fit: float) -> None:
        """Take the population's agents, buffer the metrics, and (when capturing) the frames."""
        if not self.__recording:
            return
        agents = pop.agents if isinstance(pop, Population) else list(pop)
        n_agents = len(agents)
        if n_agents == 0:
            return
        positions = np.empty((n_agents, self.n_dims), dtype=np.float64)
        fits = np.empty(n_agents, dtype=np.float64)
        for i, agent in enumerate(agents):
            positions[i, :] = agent.solution
            fits[i] = agent.fitness

        if self.capture_population:
            objectives = np.asarray([
                np.asarray(agent.objectives, dtype=np.float64).ravel() for agent in agents
            ], dtype=np.float64)
            self.__snapshots.record(positions, fits, objectives)
        self.__metrics.record(positions, fits, runtime, nfe, global_best_fit, self.__minimize)

    def finalize(self) -> None:
        """Compute the derived metrics and assemble the finished arrays."""
        if not self.__recording:
            return
        self.n_epochs = int(self.__metrics.n_epochs)
        self.__metric_matrix = self.__metrics.finalize()
        if self.capture_population:
            solution, fitness, objectives = self.__snapshots.finalize()
            self.__solution, self.__fitness, self.__objective = solution, fitness, objectives
            if self.__snapshots.n_objs is not None:
                self.n_objs = int(self.__snapshots.n_objs)
        self.__metrics.reset()
        self.__recording = False

    def __getitem__(self, key: str) -> np.ndarray:
        """Read a finished array by name: metrics first, then the population frames."""
        if self.__metric_matrix is not None:
            try:
                return self.__metric_matrix[:, METRIC_NAMES.index(key)].copy()
            except ValueError:
                pass
        if key == "solution":
            if self.__solution is not None:
                return self.__solution
        elif key == "fitness":
            if self.__fitness is not None:
                return self.__fitness
        elif key == "objectives":
            if self.__objective is not None:
                return self.__objective
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True
