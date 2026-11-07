#!/usr/bin/env python
# Created for clypto's in-memory population tracking.
# --------------------------------------------------%
"""``Tracker``: the per-run history held in ``optimizer.tracker``.

This module provides:
+ :func:`metrics` — Pydantic-style class decorator for custom per-epoch metrics.
+ :class:`DefaultMetrics` — canonical continuous metrics matching ``mealpy-lts``.
+ :class:`MetricBuffer` — the per-epoch metric values in one contiguous matrix.
+ :class:`SnapshotBuffer` — the optional population frames (solutions, fitness, objectives).
+ :class:`Tracker` — the facade holding run metadata and finished metric arrays.

Everything lives in RAM and describes the most recent ``solve()`` run. No dictionaries
are allocated in the per-epoch loop: metrics are pre-mapped to fixed column indices.
"""
import inspect
import typing

import numpy as np

CORE_METRIC_NAMES = (
    "global_best_fit",
    "current_best_fit",
    "current_worst_fit",
    "mean_fit",
    "std_fit",
    "epoch_time",
    "nfe",
    "epoch_pop_size",
)

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


class MetricDef:
    __slots__ = ("name", "func", "takes_positions", "takes_fits")

    def __init__(self, name: str, func: typing.Callable, takes_positions: bool, takes_fits: bool) -> None:
        self.name = name
        self.func = func
        self.takes_positions = takes_positions
        self.takes_fits = takes_fits


class MetricSchema:
    __slots__ = (
        "cls",
        "names",
        "col_map",
        "idx_gbest",
        "idx_cbest",
        "idx_cworst",
        "idx_mean",
        "idx_std",
        "idx_time",
        "idx_nfe",
        "idx_pop",
        "idx_diversity",
        "idx_exploration",
        "idx_exploitation",
        "custom_callers",
        "has_finalize",
    )

    def __init__(
        self,
        cls: type,
        names: tuple[str, ...],
        col_map: dict[str, int],
        idx_gbest: int,
        idx_cbest: int,
        idx_cworst: int,
        idx_mean: int,
        idx_std: int,
        idx_time: int,
        idx_nfe: int,
        idx_pop: int,
        idx_diversity: int | None,
        idx_exploration: int | None,
        idx_exploitation: int | None,
        custom_callers: tuple[tuple[int, typing.Callable, bool, bool], ...],
        has_finalize: bool,
    ) -> None:
        self.cls = cls
        self.names = names
        self.col_map = col_map
        self.idx_gbest = idx_gbest
        self.idx_cbest = idx_cbest
        self.idx_cworst = idx_cworst
        self.idx_mean = idx_mean
        self.idx_std = idx_std
        self.idx_time = idx_time
        self.idx_nfe = idx_nfe
        self.idx_pop = idx_pop
        self.idx_diversity = idx_diversity
        self.idx_exploration = idx_exploration
        self.idx_exploitation = idx_exploitation
        self.custom_callers = custom_callers
        self.has_finalize = has_finalize


def _build_metric_schema(cls: type) -> MetricSchema:
    methods: list[tuple[str, typing.Callable, bool, bool]] = []
    has_finalize = hasattr(cls, "finalize") and callable(getattr(cls, "finalize"))

    for name, method in inspect.getmembers(cls, predicate=inspect.isroutine):
        if name.startswith("_") or name == "finalize":
            continue
        sig = inspect.signature(method)
        ret = sig.return_annotation
        if ret is not inspect.Signature.empty and ret is not typing.Any:
            valid_names = ("float", "int", "np.floating", "np.integer")
            valid_types = (float, int, np.floating, np.integer)
            is_valid = ret in valid_names or ret in valid_types
            if not is_valid:
                try:
                    is_valid = isinstance(ret, type) and issubclass(ret, valid_types)
                except TypeError:
                    is_valid = False
            if not is_valid:
                raise TypeError(f"Metric method {name!r} return type must be float, got {ret!r}")

        params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
        if len(params) == 1:
            pname = params[0].name.lower()
            if "fit" in pname:
                takes_pos, takes_fit = False, True
            else:
                takes_pos, takes_fit = True, False
        elif len(params) == 2:
            takes_pos, takes_fit = True, True
        else:
            raise ValueError(f"Metric method {name!r} must accept 1 or 2 arguments ('positions', 'fits'), got {len(params)}")
        methods.append((name, method, takes_pos, takes_fit))

    # If cls is DefaultMetrics, maintain canonical METRIC_NAMES order:
    if cls is DefaultMetrics or (issubclass(cls, DefaultMetrics) and [m[0] for m in methods] == ["diversity"]):
        names = METRIC_NAMES
        col_map = {name: idx for idx, name in enumerate(names)}
        custom_callers = tuple((col_map[m[0]], m[1], m[2], m[3]) for m in methods)
        return MetricSchema(
            cls=cls,
            names=names,
            col_map=col_map,
            idx_gbest=col_map["global_best_fit"],
            idx_cbest=col_map["current_best_fit"],
            idx_cworst=col_map["current_worst_fit"],
            idx_mean=col_map["mean_fit"],
            idx_std=col_map["std_fit"],
            idx_time=col_map["epoch_time"],
            idx_nfe=col_map["nfe"],
            idx_pop=col_map["epoch_pop_size"],
            idx_diversity=col_map["diversity"],
            idx_exploration=col_map["exploration"],
            idx_exploitation=col_map["exploitation"],
            custom_callers=custom_callers,
            has_finalize=has_finalize,
        )

    # For custom classes:
    core_first = ("global_best_fit", "current_best_fit", "current_worst_fit", "mean_fit", "std_fit")
    core_last = ("epoch_time", "nfe", "epoch_pop_size")

    custom_names = [m[0] for m in methods if m[0] not in core_first and m[0] not in core_last]
    derived = []
    if issubclass(cls, DefaultMetrics) or "diversity" in custom_names:
        if "exploration" not in custom_names:
            derived.append("exploration")
        if "exploitation" not in custom_names:
            derived.append("exploitation")

    names = tuple(core_first) + tuple(custom_names) + tuple(derived) + tuple(core_last)
    col_map = {name: idx for idx, name in enumerate(names)}
    custom_callers = tuple((col_map[m[0]], m[1], m[2], m[3]) for m in methods)

    return MetricSchema(
        cls=cls,
        names=names,
        col_map=col_map,
        idx_gbest=col_map["global_best_fit"],
        idx_cbest=col_map["current_best_fit"],
        idx_cworst=col_map["current_worst_fit"],
        idx_mean=col_map["mean_fit"],
        idx_std=col_map["std_fit"],
        idx_time=col_map["epoch_time"],
        idx_nfe=col_map["nfe"],
        idx_pop=col_map["epoch_pop_size"],
        idx_diversity=col_map.get("diversity"),
        idx_exploration=col_map.get("exploration"),
        idx_exploitation=col_map.get("exploitation"),
        custom_callers=custom_callers,
        has_finalize=has_finalize,
    )


def metrics(cls=None):
    """Pydantic-style class decorator defining per-epoch metrics.

    Methods can accept ``positions: np.ndarray``, ``fits: np.ndarray``, or both,
    and must return a ``float``.
    """
    def decorate(target_cls):
        target_cls.__metric_schema__ = _build_metric_schema(target_cls)
        return target_cls

    if cls is not None:
        return decorate(cls)
    return decorate


class DefaultMetrics:
    """Canonical continuous per-epoch metrics matching mealpy-lts."""

    def diversity(self, positions: np.ndarray) -> float:
        median = np.median(positions, axis=0)
        return float(np.mean(np.abs(median - positions)))


DefaultMetrics = metrics(DefaultMetrics)


def resolve_schema(metric_spec: typing.Any = None) -> tuple[MetricSchema, typing.Any]:
    """Resolve metric class/instance/None into ``(MetricSchema, instance)``."""
    if metric_spec is None:
        return DefaultMetrics.__metric_schema__, DefaultMetrics()
    if isinstance(metric_spec, type):
        cls = metric_spec
        if not hasattr(cls, "__metric_schema__"):
            cls = metrics(cls)
        return cls.__metric_schema__, cls()
    cls = type(metric_spec)
    if not hasattr(cls, "__metric_schema__"):
        metrics(cls)
    return cls.__metric_schema__, metric_spec


class MetricBuffer:
    """Per-epoch metrics in one fixed ``(epoch, metrics)`` matrix."""

    def __init__(self) -> None:
        self.__buffer: np.ndarray | None = None
        self.__n_epochs = 0
        self.__capacity = 0
        self.__schema: MetricSchema | None = None
        self.__instance: typing.Any = None

    def allocate(self, max_epochs: int, schema: MetricSchema, instance: typing.Any = None) -> None:
        """One ``(max_epochs, n_metrics)`` matrix, ready for ``record``."""
        self.__schema = schema
        self.__instance = instance
        self.__buffer = np.empty((max(1, int(max_epochs)), len(schema.names)), dtype=np.float64)
        self.__capacity = max(1, int(max_epochs))
        self.__n_epochs = 0

    def reset(self) -> None:
        """Drop the matrix; ``allocate`` will recreate it on the next start."""
        self.__buffer = None
        self.__n_epochs = 0
        self.__capacity = 0
        self.__schema = None
        self.__instance = None

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
        if self.__schema is None or self.__buffer is None:
            return
        n = positions.shape[0]
        c_best, c_worst = (float(np.min(fits)), float(np.max(fits))) if minimize \
            else (float(np.max(fits)), float(np.min(fits)))

        schema = self.__schema
        row = self.__buffer[self.__n_epochs]
        row[schema.idx_gbest] = float(global_best_fit)
        row[schema.idx_cbest] = c_best
        row[schema.idx_cworst] = c_worst
        row[schema.idx_mean] = float(np.mean(fits))
        row[schema.idx_std] = float(np.std(fits))
        row[schema.idx_time] = float(runtime)
        row[schema.idx_nfe] = int(nfe)
        row[schema.idx_pop] = int(n)

        inst = self.__instance
        for col_idx, func, takes_pos, takes_fit in schema.custom_callers:
            if takes_pos and takes_fit:
                row[col_idx] = float(func(inst, positions, fits))
            elif takes_pos:
                row[col_idx] = float(func(inst, positions))
            else:
                row[col_idx] = float(func(inst, fits))

        self.__n_epochs += 1

    def finalize(self) -> np.ndarray:
        """Derive exploration/exploitation and trim to ``n_epochs`` fresh columns."""
        if self.__schema is None or self.__buffer is None:
            return np.empty((0, 0), dtype=np.float64)
        schema = self.__schema
        if schema.idx_diversity is not None and schema.idx_exploration is not None and schema.idx_exploitation is not None:
            diversity = self.__buffer[: self.__n_epochs, schema.idx_diversity].copy()
            if diversity.size and np.max(diversity) > 0:
                exploration = 100.0 * diversity / np.max(diversity)
            else:
                exploration = np.zeros_like(diversity)
            self.__buffer[: self.__n_epochs, schema.idx_exploration] = exploration
            self.__buffer[: self.__n_epochs, schema.idx_exploitation] = 100.0 - exploration

        data = self.__buffer[: self.__n_epochs].copy()
        if schema.has_finalize and self.__instance is not None:
            self.__instance.finalize(data, schema.col_map)
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
    """The run history of the engine: per-epoch metrics and optional population frames."""

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
        self.__schema: MetricSchema | None = None
        self.__col_map: dict[str, int] | None = None
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
        self.__schema = None
        self.__col_map = None
        self.__solution = self.__fitness = self.__objective = None

    @property
    def recording(self) -> bool:
        return self.__recording

    @property
    def metric_names(self) -> tuple[str, ...]:
        return self.__schema.names if self.__schema is not None else ()

    def start(self, optimizer, problem, seed=None, capture_population: bool = False, metrics: typing.Any = None) -> None:
        """Begin a new recording with the specified metric schema."""
        self.reset()
        schema, instance = resolve_schema(metrics)
        self.__schema = schema
        self.__col_map = schema.col_map
        self.__minimize = str(problem.sense) == "min"
        self.capture_population = bool(capture_population)
        self.optimizer_name = str(optimizer.name)
        self.problem_name = str(problem.name)
        self.seed = seed
        self.pop_size = int(optimizer.pop_size)
        self.n_dims = int(problem.n_dims)
        self.sense = str(problem.sense)
        self.__metrics.allocate(optimizer.epoch, schema, instance)
        self.__recording = True

    def record(self, pop, runtime: float, nfe: int, global_best_fit: float) -> None:
        """Take the population's agents, buffer the metrics, and (when capturing) the frames."""
        if not self.__recording:
            return
        agents = list(pop.agents) if hasattr(pop, "agents") else list(pop)
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
        if self.__metric_matrix is not None and self.__col_map is not None and key in self.__col_map:
            return self.__metric_matrix[:, self.__col_map[key]].copy()
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
        if self.__metric_matrix is not None and self.__col_map is not None and key in self.__col_map:
            return True
        if key in ("solution", "fitness", "objectives"):
            try:
                self[key]
                return True
            except KeyError:
                return False
        return False
