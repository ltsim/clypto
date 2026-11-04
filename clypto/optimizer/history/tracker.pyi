import numpy as np
import typing

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
    meta: dict[str, typing.Any]
    def __init__(self) -> None: ...
    def reset(self) -> None:
        """Discard everything from the previous run and return to the empty state."""
    @property
    def recording(self) -> bool: ...
    def start(self, optimizer: typing.Any, problem: typing.Any, seed: int | None = ..., capture_population: bool = ...) -> None:
        """Begin a new recording and describe the run through ``meta``."""
    def record(self, pop: typing.Any, runtime: float, nfe: int, global_best_fit: float) -> None:
        """Buffer the current epoch's population snapshot (if enabled) and metrics."""
    def finalize(self) -> None:
        """Compute derived metrics, assemble the numpy arrays, and close the run."""
    def __getitem__(self, key: str) -> np.ndarray:
        """Read a finished array by name (metrics and population snapshots alike)."""
