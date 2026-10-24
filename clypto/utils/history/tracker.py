#!/usr/bin/env python
# Created for clypto's in-memory population tracking.
# --------------------------------------------------%
import typing

import numpy as np

_METRIC_NAMES = (
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


def _pad_stack(frames: list[np.ndarray]) -> np.ndarray:
    """Stack per-epoch ``(rows, ...)`` frames into ``(epochs, max_rows, ...)`` with NaN padding."""
    max_rows = max(frame.shape[0] for frame in frames)
    out = np.full((len(frames), max_rows) + frames[0].shape[1:], np.nan, dtype=np.float64)
    for idx, frame in enumerate(frames):
        out[idx, : frame.shape[0]] = frame
    return out


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
        self.meta: dict[str, typing.Any] = {}
        self._data: dict[str, np.ndarray] = {}
        self._metrics: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
        self._solution_frames: list[np.ndarray] = []
        self._fitness_frames: list[np.ndarray] = []
        self._objective_frames: list[np.ndarray] = []
        self._recording = False
        self._capture_population = False
        self._n_objs: typing.Optional[int] = None
        self._minmax = "min"

    def reset(self) -> None:
        """Discard everything from the previous run and return to the empty state."""
        self.meta = {}
        self._data = {}
        self._metrics = {name: [] for name in _METRIC_NAMES}
        self._solution_frames = []
        self._fitness_frames = []
        self._objective_frames = []
        self._recording = False
        self._capture_population = False
        self._n_objs = None
        self._minmax = "min"

    @property
    def recording(self) -> bool:
        return self._recording

    def start(
        self,
        optimizer,
        problem,
        seed: typing.Optional[int] = None,
        capture_population: bool = False,
    ) -> None:
        """Begin a new recording and describe the run through ``meta``."""
        self.reset()
        self._capture_population = bool(capture_population)
        self._minmax = problem.minmax

        self.meta = {
            "optimizer": str(optimizer.name),
            "problem": str(problem.get_name()),
            "seed": seed,
            "pop_size": int(optimizer.pop_size),
            "n_dims": int(problem.n_dims),
            "minmax": str(self._minmax),
            "capture_population": self._capture_population,
        }
        self._recording = True

    def record(
        self,
        pop: list,
        runtime: float,
        nfe: int,
        global_best_fit: float,
    ) -> None:
        """Buffer the current epoch's population snapshot (if enabled) and metrics."""
        if not self._recording:
            return

        positions = self._solutions(pop)
        fits = self._fitnesses(pop)

        if self._capture_population:
            self._solution_frames.append(positions)
            self._fitness_frames.append(fits)
            if self._n_objs is None:
                self._n_objs = int(np.asarray(pop[0].target.objectives).ravel().size)
            if self._n_objs > 1:
                self._objective_frames.append(self._objectives(pop))

        if self._minmax == "min":
            c_best, c_worst = float(np.min(fits)), float(np.max(fits))
        else:
            c_best, c_worst = float(np.max(fits)), float(np.min(fits))
        diversity = float(np.mean(np.abs(np.median(positions, axis=0) - positions)))

        metrics = self._metrics
        metrics["global_best_fit"].append(float(global_best_fit))
        metrics["current_best_fit"].append(c_best)
        metrics["current_worst_fit"].append(c_worst)
        metrics["mean_fit"].append(float(np.mean(fits)))
        metrics["std_fit"].append(float(np.std(fits)))
        metrics["diversity"].append(diversity)
        metrics["epoch_time"].append(float(runtime))
        metrics["nfe"].append(int(nfe))
        metrics["epoch_pop_size"].append(int(len(pop)))

    def finalize(self) -> None:
        """Compute derived metrics, assemble the numpy arrays, and close the run."""
        if not self._recording:
            return

        n_epochs = len(self._metrics["nfe"])
        self.meta["n_epochs"] = int(n_epochs)

        diversity = np.asarray(self._metrics["diversity"], dtype=np.float64)
        if diversity.size and np.max(diversity) > 0:
            exploration = 100.0 * diversity / np.max(diversity)
        else:
            exploration = np.zeros_like(diversity)
        self._metrics["exploration"] = exploration.tolist()
        self._metrics["exploitation"] = (100.0 - exploration).tolist()

        self._data = {
            name: np.asarray(values, dtype=np.float64) for name, values in self._metrics.items()
        }

        if self._capture_population:
            if self._solution_frames:
                self._data["solution"] = _pad_stack(self._solution_frames)
                self._data["fitness"] = _pad_stack(self._fitness_frames)
            if self._objective_frames:
                self._data["objectives"] = _pad_stack(self._objective_frames)
            if self._n_objs is not None:
                self.meta["n_objs"] = int(self._n_objs)

        self._solution_frames = []
        self._fitness_frames = []
        self._objective_frames = []
        self._recording = False

    def __getitem__(self, key: str) -> np.ndarray:
        """Read a finished array by name (metrics and population snapshots alike)."""
        if key in self._data:
            return self._data[key]
        raise KeyError(key)

    @staticmethod
    def _solutions(pop: list) -> np.ndarray:
        return np.asarray([agent.solution for agent in pop], dtype=np.float64)

    @staticmethod
    def _fitnesses(pop: list) -> np.ndarray:
        return np.asarray([agent.target.fitness for agent in pop], dtype=np.float64)

    @staticmethod
    def _objectives(pop: list) -> np.ndarray:
        return np.asarray(
            [np.asarray(agent.target.objectives).ravel() for agent in pop],
            dtype=np.float64,
        )
