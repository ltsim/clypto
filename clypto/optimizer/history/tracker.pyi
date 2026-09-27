import numpy as np
import typing
from _typeshed import Incomplete

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
    before: Incomplete
    after: Incomplete
    epoch: int | None
    path: str | None
    def __init__(self, before: typing.Callable | None = None, after: typing.Callable | None = None) -> None: ...
    def on_before(self, fn: typing.Callable) -> typing.Callable:
        """Register ``fn`` as the pre-iteration hook (usable as a decorator)."""
    def on_after(self, fn: typing.Callable) -> typing.Callable:
        """Register ``fn`` as the post-iteration hook (usable as a decorator)."""
    @property
    def group(self):
        """The underlying Zarr group (``None`` until :meth:`start` is called)."""
    @property
    def recording(self) -> bool: ...
    def start(self, optimizer, problem, seed: int | None = None, capture_population: bool = False, history_path: str | None = None) -> None:
        """Open the store and prepare the arrays for a new run."""
    def record(self, epoch: int, pop: list, runtime: float, nfe: int, global_best_fit: float) -> None:
        """Write the current epoch's population (if enabled) and metrics."""
    def finalize(self) -> None:
        """Compute derived metrics, trim unused capacity, and flush the store."""
    def __getitem__(self, key: str) -> np.ndarray:
        """Read an array by name; metric names resolve under the ``metrics`` group."""
