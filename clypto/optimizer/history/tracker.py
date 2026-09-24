#!/usr/bin/env python
# Created for clypto's Zarr-backed population tracking.
# --------------------------------------------------%
import typing

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore, MemoryStore

_POPULATION_ARRAYS = ("solution", "fitness", "objectives")
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
_COMPRESSORS = [BloscCodec(cname="zstd", clevel=3)]


class Tracker:
    """
    Record per-epoch metrics and (optionally) full population snapshots into a Zarr store.

    Notes
    ~~~~~
    + Metrics are small ``(epoch,)`` arrays; full population snapshots are chunked
      ``(epoch, pop_size, n_dims)`` arrays streamed one epoch at a time, so RAM stays flat.
    + If no store path is given the data lives in an in-memory Zarr store owned by this
      object; pass ``history_path`` to persist it on disk.
    + Attach custom per-iteration callbacks with ``tracker.before`` / ``tracker.after``
      (or the ``on_before`` / ``on_after`` decorators)::

            @optimizer.tracker.on_after
            def after_iteration(population):
                ...

    The current epoch is available as ``tracker.epoch`` while a hook runs.
    """

    def __init__(
        self,
        before: typing.Optional[typing.Callable] = None,
        after: typing.Optional[typing.Callable] = None,
    ) -> None:
        self.before = before
        self.after = after
        self.epoch: typing.Optional[int] = None
        self.path: typing.Optional[str] = None

        self._store = None
        self._group = None
        self._optimizer = None
        self._recording = False
        self._capture_population = False
        self._n_dims = 0
        self._n_objs: typing.Optional[int] = None
        self._minmax = "min"
        self._max_pop = 0
        self._epoch_capacity = 0
        self._n_recorded = 0
        self._metrics: dict[str, list] = {name: [] for name in _METRIC_NAMES}

    def on_before(self, fn: typing.Callable) -> typing.Callable:
        """Register ``fn`` as the pre-iteration hook (usable as a decorator)."""
        self.before = fn
        return fn

    def on_after(self, fn: typing.Callable) -> typing.Callable:
        """Register ``fn`` as the post-iteration hook (usable as a decorator)."""
        self.after = fn
        return fn

    @property
    def group(self):
        """The underlying Zarr group (``None`` until :meth:`start` is called)."""
        return self._group

    @property
    def recording(self) -> bool:
        return self._recording

    def start(
        self,
        optimizer,
        problem,
        seed: typing.Optional[int] = None,
        capture_population: bool = False,
        history_path: typing.Optional[str] = None,
    ) -> None:
        """Open the store and prepare the arrays for a new run."""
        self._optimizer = optimizer
        self.path = history_path
        self._store = LocalStore(history_path) if history_path else MemoryStore()
        self._group = zarr.open_group(store=self._store, mode="w")

        self._capture_population = bool(capture_population)
        self._n_dims = int(problem.n_dims)
        self._n_objs = None
        self._minmax = problem.minmax
        self._max_pop = max(int(optimizer.pop_size), 1)
        self._epoch_capacity = max(int(optimizer.epoch or 1), 1)
        self._n_recorded = 0
        self._metrics = {name: [] for name in _METRIC_NAMES}

        self._group.attrs.update(
            {
                "optimizer": str(optimizer.name),
                "problem": str(problem.get_name()),
                "seed": seed,
                "pop_size": int(optimizer.pop_size),
                "n_dims": self._n_dims,
                "minmax": str(self._minmax),
                "capture_population": self._capture_population,
            }
        )

        if self._capture_population:
            self._create_population_arrays()

        self._recording = True

    def record(
        self,
        epoch: int,
        pop: list,
        runtime: float,
        nfe: int,
        global_best_fit: float,
    ) -> None:
        """Write the current epoch's population (if enabled) and metrics."""
        if not self._recording:
            return

        idx = self._n_recorded
        n_agents = len(pop)

        if self._capture_population:
            self._grow_epoch(idx)
            if n_agents > self._max_pop:
                self._grow_pop(n_agents)
            self._group["solution"][idx, :n_agents] = self._solutions(pop)
            self._group["fitness"][idx, :n_agents] = self._fitnesses(pop)
            if self._n_objs is None:
                self._n_objs = int(np.asarray(pop[0].target.objectives).ravel().size)
                if self._n_objs > 1:
                    self._create_objectives_array()
            if self._n_objs > 1:
                self._group["objectives"][idx, :n_agents] = self._objectives(pop)

        fits = self._fitnesses(pop)
        if self._minmax == "min":
            c_best, c_worst = float(np.min(fits)), float(np.max(fits))
        else:
            c_best, c_worst = float(np.max(fits)), float(np.min(fits))
        positions = self._solutions(pop)
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
        metrics["epoch_pop_size"].append(int(n_agents))

        self._n_recorded += 1

    def finalize(self) -> None:
        """Compute derived metrics, trim unused capacity, and flush the store."""
        if not self._recording:
            return

        n_epochs = self._n_recorded

        if self._capture_population:
            for name in self._population_array_names():
                arr = self._group[name]
                arr.resize((n_epochs,) + arr.shape[1:])

        diversity = np.asarray(self._metrics["diversity"], dtype=np.float64)
        if diversity.size and np.max(diversity) > 0:
            exploration = 100.0 * diversity / np.max(diversity)
        else:
            exploration = np.zeros_like(diversity)
        self._metrics["exploration"] = exploration.tolist()
        self._metrics["exploitation"] = (100.0 - exploration).tolist()

        metrics_group = self._group.create_group("metrics")
        for name in _METRIC_NAMES:
            arr = metrics_group.create_array(
                name,
                shape=(n_epochs,),
                chunks=(max(n_epochs, 1),),
                dtype="float64",
                fill_value=np.nan,
            )
            arr[:] = np.asarray(self._metrics[name], dtype=np.float64)

        self._group.attrs["n_epochs"] = int(n_epochs)
        if self._n_objs is not None:
            self._group.attrs["n_objs"] = int(self._n_objs)
        self._recording = False

    def __getitem__(self, key: str) -> np.ndarray:
        """Read an array by name; metric names resolve under the ``metrics`` group."""
        if self._group is None:
            raise KeyError("Tracker has not been started.")
        if key in self._group:
            return self._group[key][:]
        if "metrics" in self._group and key in self._group["metrics"]:
            return self._group["metrics"][key][:]
        raise KeyError(key)

    def _population_array_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in _POPULATION_ARRAYS if name in self._group
        )

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

    def _create_population_arrays(self) -> None:
        self._group.create_array(
            "solution",
            shape=(self._epoch_capacity, self._max_pop, self._n_dims),
            chunks=(1, self._max_pop, self._n_dims),
            dtype="float64",
            fill_value=np.nan,
            compressors=_COMPRESSORS,
        )
        self._group.create_array(
            "fitness",
            shape=(self._epoch_capacity, self._max_pop),
            chunks=(1, self._max_pop),
            dtype="float64",
            fill_value=np.nan,
            compressors=_COMPRESSORS,
        )

    def _create_objectives_array(self) -> None:
        self._group.create_array(
            "objectives",
            shape=(self._epoch_capacity, self._max_pop, int(self._n_objs)),
            chunks=(1, self._max_pop, int(self._n_objs)),
            dtype="float64",
            fill_value=np.nan,
            compressors=_COMPRESSORS,
        )

    def _grow_epoch(self, idx: int) -> None:
        if idx < self._epoch_capacity:
            return
        new_capacity = max(idx + 1, self._epoch_capacity * 2)
        for name in self._population_array_names():
            arr = self._group[name]
            arr.resize((new_capacity,) + arr.shape[1:])
        self._epoch_capacity = new_capacity

    def _grow_pop(self, n_agents: int) -> None:
        for name in self._population_array_names():
            arr = self._group[name]
            shape = list(arr.shape)
            shape[1] = n_agents
            arr.resize(tuple(shape))
        self._max_pop = n_agents
