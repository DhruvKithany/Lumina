"""
Moving-average smoothing buffer for landmark coordinates.

Maintains a sliding window of recent values and returns their mean.
Supports both scalar and multi-dimensional (e.g. [x, y]) inputs via numpy.
"""

from __future__ import annotations

from collections import deque
from typing import Union

import numpy as np


class SmoothingBuffer:
    """
    Fixed-size deque that returns the running mean of buffered values.
    Works with scalars and numpy arrays (e.g. landmark [x, y] pairs).
    """

    def __init__(self, window_size: int = 10) -> None:
        self.buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, new_value: Union[float, list[float], np.ndarray]) -> np.ndarray:
        """
        Append a value and return the smoothed (mean) result.
        Accepts a scalar, list, or numpy array.
        """
        arr = np.asarray(new_value, dtype=np.float64)
        self.buffer.append(arr)
        return np.mean(self.buffer, axis=0)

    def reset(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    @property
    def ready(self) -> bool:
        """True when the buffer has at least one sample."""
        return len(self.buffer) > 0