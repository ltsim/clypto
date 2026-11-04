import numpy as np

METRIC_NAMES: tuple[str, ...]

class MetricStore:
    """
    Buffer one numeric value per metric per epoch and derive the rest.

    Notes
    ~~~~~
    + :meth:`record` computes the whole epoch at once from ``(pop, n_dims)``
      positions and ``(pop,)`` fitnesses, so no agent or population object is
      involved here.
    + ``exploration`` and ``exploitation`` are not recorded per epoch: they are
      derived from ``diversity`` in :meth:`finalize`, rescaled to 0–100.
    + :meth:`finalize` returns fresh numpy arrays; the store keeps its buffers so
      a caller can inspect it, but the tracker resets it before the next run.
    """

    def __init__(self) -> None: ...
    def reset(self) -> None:
        """Drop every buffered epoch."""
    @property
    def n_epochs(self) -> int:
        """Number of recorded epochs."""
    def record(self, positions: np.ndarray, fits: np.ndarray, runtime: float, nfe: int, global_best_fit: float, sense: str) -> None:
        """Append one epoch's statistics computed from the population arrays."""
    def finalize(self) -> dict[str, np.ndarray]:
        """Derive exploration/exploitation and freeze every buffer into an array."""
