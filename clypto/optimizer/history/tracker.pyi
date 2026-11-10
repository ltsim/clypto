import typing

import numpy as np

CORE_METRIC_NAMES: tuple[str, ...]
METRIC_NAMES: tuple[str, ...]

_C = typing.TypeVar("_C")

@typing.overload
def metrics(cls: _C) -> _C: ...
@typing.overload
def metrics() -> typing.Callable[[_C], _C]: ...

class DefaultMetrics:
    """Canonical continuous per-epoch metrics matching mealpy-lts."""
    def diversity(self, positions: np.ndarray) -> float: ...

class MetricDef:
    name: str
    func: typing.Callable
    takes_positions: bool
    takes_fits: bool

class MetricSchema:
    cls: type
    names: tuple[str, ...]
    col_map: dict[str, int]
    idx_gbest: int
    idx_cbest: int
    idx_cworst: int
    idx_mean: int
    idx_std: int
    idx_time: int
    idx_nfe: int
    idx_pop: int
    idx_diversity: int | None
    idx_exploration: int | None
    idx_exploitation: int | None
    custom_callers: tuple[tuple[int, typing.Callable, bool, bool], ...]
    has_finalize: bool

def resolve_schema(metric_spec: typing.Any = ...) -> tuple[MetricSchema, typing.Any]: ...

class MetricBuffer:
    """Per-epoch metrics in one fixed ``(epoch, metrics)`` matrix."""

    def __init__(self) -> None: ...
    def allocate(self, max_epochs: int, schema: MetricSchema, instance: typing.Any = ...) -> None:
        """One ``(max_epochs, n_metrics)`` matrix, ready for ``record``."""
    def reset(self) -> None:
        """Drop the matrix; ``allocate`` will recreate it on the next start."""
    @property
    def n_epochs(self) -> int: ...
    @property
    def buffer(self) -> np.ndarray | None: ...
    def record(self, positions: np.ndarray, fits: np.ndarray, runtime: float, nfe: int,
               global_best_fit: float, minimize: bool = ...) -> None:
        """Append one epoch's statistics computed from the population arrays."""
    def finalize(self) -> np.ndarray:
        """Derive exploration/exploitation and trim to ``n_epochs`` fresh columns."""


class SnapshotBuffer:
    """Population frames per epoch, one NaN-padded array per capture field."""

    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    @property
    def n_objs(self) -> int | None: ...
    def record(self, positions: np.ndarray, fits: np.ndarray, objectives: np.ndarray | None = ...) -> None:
        """Append one epoch's frames (positions, fitness, objectives when multi-objective)."""
    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """``(solution, fitness, objectives)`` arrays with exact shapes."""


class Tracker:
    """The run history of the engine: per-epoch metrics and optional population frames."""

    optimizer_name: str
    problem_name: str
    seed: int | None
    pop_size: int
    n_dims: int
    sense: str
    capture_population: bool
    n_epochs: int
    n_objs: int

    def __init__(self) -> None: ...
    def reset(self) -> None:
        """Discard everything from the previous run and return to the initial state."""
    @property
    def recording(self) -> bool: ...
    @property
    def metric_names(self) -> tuple[str, ...]: ...
    def start(self, optimizer: typing.Any, problem: typing.Any, seed: int | None = ...,
              capture_population: bool = ..., metrics: typing.Any = ...) -> None:
        """Begin a new recording; run metadata sets the typed fields."""
    def record(self, pop: typing.Any, runtime: float, nfe: int, global_best_fit: float) -> None:
        """Take the population's agents, buffer the metrics, and the frames when capturing."""
    def finalize(self) -> None:
        """Compute the derived metrics and assemble the finished arrays."""
    def __getitem__(self, key: str) -> np.ndarray:
        """Read a finished array by name: metrics first, then the population frames."""
    def __contains__(self, key: str) -> bool: ...
