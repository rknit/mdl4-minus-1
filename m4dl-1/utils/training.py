from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import tensorflow as tf

@dataclass
class RunningConfusion:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    correct: int = 0

    def update(self, y_true_idx: int, y_pred_idx: int, pos_idx: int = 0) -> None:
        if y_true_idx == y_pred_idx:
            self.correct += 1
        if y_true_idx == pos_idx and y_pred_idx == pos_idx:
            self.tp += 1
        elif y_true_idx != pos_idx and y_pred_idx == pos_idx:
            self.fp += 1
        elif y_true_idx == pos_idx and y_pred_idx != pos_idx:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


def make_tf_dataset(
    x: np.ndarray | tf.Tensor,
    y: np.ndarray | tf.Tensor,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: Optional[int] = None,
    buffer_size: Optional[int] = None,
) -> tf.data.Dataset:
    """Keras-like dataset builder. Shuffles only when requested."""
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x)
    if not isinstance(y, tf.Tensor):
        y = tf.convert_to_tensor(y)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        n = int(x.shape[0]) # type: ignore[attr-defined]
        buf = buffer_size or n  # perfect shuffle for small tabular datasets
        ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
