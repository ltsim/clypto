import typing

import numpy as np

METRIC_NAMES: tuple[str, ...]

class MetricBuffer:
    """Per-epoch metrics in one fixed ``(epoch, metrics)`` matrix."""

    def __init__(self) -> None: ...
    def allocate(self, max_epochs: int) -> None:
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
    """The run history of the engine: per-epoch metrics and optional population frames.

    ``array["global_best_fit"]`` and friends are read by name; the run's metadata
    lives on typed fields (``optimizer_name``, ``problem_name``, ``seed``,
    ``pop_size``, ``n_dims``, ``sense``, ``capture_population``, ``n_epochs``,
    ``n_objs``). ``tracker.meta`` is gone.
    """

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
    def start(self, optimizer: typing.Any, problem: typing.Any, seed: int | None = ...,
              capture_population: bool = ...) -> None:
        """Begin a new recording; run metadata sets the typed fields."""
    def record(self, pop: typing.Any, runtime: float, nfe: int, global_best_fit: float) -> None:
        """Take the population's agents, buffer the metrics, and the frames when capturing."""
    def finalize(self) -> None:
        """Compute the derived metrics and assemble the finished arrays."""
    def __getitem__(self, key: str) -> np.ndarray:
        """Read a finished array by name: metrics first, then the population frames."""
