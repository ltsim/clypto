#!/usr/bin/env python
# Created for clypto's per-epoch population snapshots.
# --------------------------------------------------%
"""``SnapshotBuffer``: per-epoch population frames, stacked with NaN padding.

Optimizers may grow and shrink their population from epoch to epoch (SAP_DE,
...), so frames cannot be pre-allocated. Each :meth:`SnapshotBuffer.record`
appends one ``(pop, ...)`` frame and :meth:`SnapshotBuffer.finalize` stacks them
into ``(epochs, max_pop, ...)``, padding the shorter epochs with NaN.
"""
import numpy as np


def _pad_stack(frames: list) -> np.ndarray:
    """Stack per-epoch ``(rows, ...)`` frames into ``(epochs, max_rows, ...)`` with NaN padding."""
    max_rows = max(frame.shape[0] for frame in frames)
    out = np.full((len(frames), max_rows) + frames[0].shape[1:], np.nan, dtype=np.float64)
    for idx, frame in enumerate(frames):
        out[idx, : frame.shape[0]] = frame
    return out


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

    def __init__(self) -> None:
        self.__capture = False
        self.__solution_frames: list = []
        self.__fitness_frames: list = []
        self.__objective_frames: list = []
        self.__n_objs = None

    def reset(self) -> None:
        """Drop every buffered frame and forget the objective count."""
        self.__solution_frames = []
        self.__fitness_frames = []
        self.__objective_frames = []
        self.__n_objs = None

    @property
    def capture(self) -> bool:
        """Whether frames are being buffered at all."""
        return self.__capture

    @capture.setter
    def capture(self, value: bool) -> None:
        self.__capture = bool(value)

    @property
    def n_objs(self) -> int | None:
        """Number of objectives seen, or ``None`` before the first frame."""
        return self.__n_objs

    def record(
        self,
        positions: np.ndarray,
        fits: np.ndarray,
        objectives: np.ndarray | None = None,
    ) -> None:
        """Append one epoch's frames (positions, fitnesses, and objectives when multi-objective)."""
        if not self.__capture:
            return

        self.__solution_frames.append(positions)
        self.__fitness_frames.append(fits)
        if self.__n_objs is None:
            self.__n_objs = int(objectives.shape[1])
        if self.__n_objs > 1:
            self.__objective_frames.append(objectives)

    def finalize(self) -> dict:
        """Stack the buffered frames into NaN-padded arrays and release them."""
        if not self.__capture:
            return {}

        data = {}
        if self.__solution_frames:
            data["solution"] = _pad_stack(self.__solution_frames)
            data["fitness"] = _pad_stack(self.__fitness_frames)
        if self.__objective_frames:
            data["objectives"] = _pad_stack(self.__objective_frames)

        self.__solution_frames = []
        self.__fitness_frames = []
        self.__objective_frames = []
        return data
