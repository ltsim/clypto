#!/usr/bin/env python
# Created for clypto's per-epoch metric history.
# --------------------------------------------------%
"""``MetricStore``: the per-epoch metric buffers and the values derived from them.

:class:`~clypto.optimizer.history.tracker.Tracker` delegates the numeric work
here. Each :meth:`MetricStore.record` appends one epoch's statistics computed
from the population arrays, and :meth:`MetricStore.finalize` turns the buffers
into the ``(epoch,)`` numpy arrays ``tracker[key]`` returns — including the
``exploration``/``exploitation`` pair derived from ``diversity``.
"""
import numpy as np

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

    def __init__(self) -> None:
        self.__buffers: dict = {}
        self.reset()

    def reset(self) -> None:
        """Drop every buffered epoch."""
        self.__buffers = {name: [] for name in METRIC_NAMES}

    @property
    def n_epochs(self) -> int:
        """Number of recorded epochs."""
        return len(self.__buffers["nfe"])

    def record(
        self,
        positions: np.ndarray,
        fits: np.ndarray,
        runtime: float,
        nfe: int,
        global_best_fit: float,
        sense: str,
    ) -> None:
        """Append one epoch's statistics computed from the population arrays."""
        if sense == "min":
            c_best, c_worst = float(np.min(fits)), float(np.max(fits))
        else:
            c_best, c_worst = float(np.max(fits)), float(np.min(fits))
        diversity = float(np.mean(np.abs(np.median(positions, axis=0) - positions)))

        buffers = self.__buffers
        buffers["global_best_fit"].append(float(global_best_fit))
        buffers["current_best_fit"].append(c_best)
        buffers["current_worst_fit"].append(c_worst)
        buffers["mean_fit"].append(float(np.mean(fits)))
        buffers["std_fit"].append(float(np.std(fits)))
        buffers["diversity"].append(diversity)
        buffers["epoch_time"].append(float(runtime))
        buffers["nfe"].append(int(nfe))
        buffers["epoch_pop_size"].append(int(positions.shape[0]))

    def finalize(self) -> dict:
        """Derive exploration/exploitation and freeze every buffer into an array."""
        diversity = np.asarray(self.__buffers["diversity"], dtype=np.float64)
        if diversity.size and np.max(diversity) > 0:
            exploration = 100.0 * diversity / np.max(diversity)
        else:
            exploration = np.zeros_like(diversity)

        data = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in self.__buffers.items()
        }
        data["exploration"] = exploration
        data["exploitation"] = 100.0 - exploration
        return data
