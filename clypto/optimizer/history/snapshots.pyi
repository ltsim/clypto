import numpy as np

def _pad_stack(frames: list) -> np.ndarray:
    """Stack per-epoch ``(rows, ...)`` frames into ``(epochs, max_rows, ...)`` with NaN padding."""

class SnapshotBuffer:
    """
    Buffer one population snapshot per epoch and assemble the padded arrays.

    Notes
    ~~~~~
    + :meth:`record` is a no-op unless :attr:`capture` is on, so an untracked
      population costs nothing.
    + ``objectives`` frames are only kept for a multi-objective problem; the
      first recorded frame decides ``n_objs``.
    + :meth:`finalize` releases the buffered frames once they are stacked, so the
      padded arrays are the only copy left in memory.
    """

    def __init__(self) -> None: ...
    def reset(self) -> None:
        """Drop every buffered frame and forget the objective count."""
    @property
    def capture(self) -> bool:
        """Whether frames are being buffered at all."""
    @capture.setter
    def capture(self, value: bool) -> None: ...
    @property
    def n_objs(self) -> int | None:
        """Number of objectives seen, or ``None`` before the first frame."""
    def record(self, positions: np.ndarray, fits: np.ndarray, objectives: np.ndarray | None = ...) -> None:
        """Append one epoch's frames (positions, fitnesses, and objectives when multi-objective)."""
    def finalize(self) -> dict[str, np.ndarray]:
        """Stack the buffered frames into NaN-padded arrays and release them."""
