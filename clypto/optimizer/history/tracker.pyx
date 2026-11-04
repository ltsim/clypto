#!/usr/bin/env python
# Created for clypto's in-memory population tracking.
# --------------------------------------------------%
"""``Tracker``: the per-run history held in ``optimizer.tracker``.

The tracker is the facade the engine talks to. It owns the run metadata and
delegates the numeric work: :class:`~clypto.optimizer.history.metrics.MetricStore`
buffers the per-epoch metrics (and derives exploration/exploitation), while
:class:`~clypto.optimizer.history.snapshots.SnapshotBuffer` buffers the optional
population snapshots, stacked NaN-padded to the widest epoch when the run closes.
Everything lives in RAM and describes the most recent ``solve()`` run. Read
arrays by name (``tracker["global_best_fit"]``) and run metadata through
``tracker.meta``.
"""
import typing

import numpy as np

from clypto.optimizer.history.metrics import MetricStore
from clypto.optimizer.history.snapshots import SnapshotBuffer


class Tracker:
    """
    Record per-epoch metrics and (optionally) full population snapshots in memory.

    Notes
    ~~~~~
    + Metrics are small ``(epoch,)`` numpy arrays; population snapshots are buffered
      one ``(pop_size, n_dims)`` frame per epoch and stacked, NaN-padded to the
      widest epoch, when :meth:`finalize` runs.
    + Everything lives in RAM and belongs to the most recent ``solve`` run; a
      ``solve`` that does not track leaves an empty tracker behind.
    + Read arrays by name: ``tracker["global_best_fit"]``, ``tracker["solution"]``;
      run metadata (optimizer, problem, seed, ``n_epochs``, ...) lives in
      ``tracker.meta``.

    Per-iteration callbacks are not part of the tracker — pass them to
    ``solve(before_iteration=..., after_iteration=...)`` instead.
    """

    def __init__(self) -> None:
        self.meta: dict = {}
        self.__data: dict = {}
        self.__metrics = MetricStore()
        self.__snapshots = SnapshotBuffer()
        self.__recording = False

    def reset(self) -> None:
        """Discard everything from the previous run and return to the empty state."""
        self.meta = {}
        self.__data = {}
        self.__metrics.reset()
        self.__snapshots.reset()
        self.__recording = False

    @property
    def recording(self) -> bool:
        return self.__recording

    def start(
        self,
        optimizer,
        problem,
        seed: typing.Optional[int] = None,
        capture_population: bool = False,
    ) -> None:
        """Begin a new recording and describe the run through ``meta``."""
        self.reset()
        self.__snapshots.capture = bool(capture_population)

        self.meta = {
            "optimizer": str(optimizer.name),
            "problem": str(problem.name),
            "seed": seed,
            "pop_size": int(optimizer.pop_size),
            "n_dims": int(problem.n_dims),
            "sense": str(problem.sense),
            "capture_population": self.__snapshots.capture,
        }
        self.__recording = True

    def record(
        self,
        pop,
        runtime: float,
        nfe: int,
        global_best_fit: float,
    ) -> None:
        """Buffer the current epoch's population snapshot (if enabled) and metrics."""
        if not self.__recording:
            return

        positions = self._solutions(pop)
        fits = self._fitnesses(pop)
        # Objectives are only needed for a captured population; avoid the conversion otherwise.
        objectives = self._objectives(pop) if self.__snapshots.capture else None

        self.__snapshots.record(positions, fits, objectives)
        self.__metrics.record(
            positions, fits, runtime, nfe, global_best_fit, self.meta["sense"]
        )

    def finalize(self) -> None:
        """Compute derived metrics, assemble the numpy arrays, and close the run."""
        if not self.__recording:
            return

        self.meta["n_epochs"] = int(self.__metrics.n_epochs)
        self.__data = self.__metrics.finalize()

        if self.__snapshots.capture:
            self.__data.update(self.__snapshots.finalize())
            if self.__snapshots.n_objs is not None:
                self.meta["n_objs"] = int(self.__snapshots.n_objs)

        self.__recording = False

    def __getitem__(self, key: str) -> np.ndarray:
        """Read a finished array by name (metrics and population snapshots alike)."""
        if key in self.__data:
            return self.__data[key]
        raise KeyError(key)

    # A Population iterates its agents; plain lists of agents are accepted too.
    @staticmethod
    def _solutions(pop) -> np.ndarray:
        return np.asarray([agent.solution for agent in pop], dtype=np.float64)

    @staticmethod
    def _fitnesses(pop) -> np.ndarray:
        return np.asarray([agent.fitness for agent in pop], dtype=np.float64)

    @staticmethod
    def _objectives(pop) -> np.ndarray:
        return np.asarray(
            [np.asarray(agent.objectives).ravel() for agent in pop],
            dtype=np.float64,
        )
