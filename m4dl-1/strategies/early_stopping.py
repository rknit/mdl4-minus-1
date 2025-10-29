from .base import EarlyStoppingStrategy
from typing import List


class CallbackEarlyStoppingStrategy(EarlyStoppingStrategy):
    """Keras early stopping with validation split."""

    def __init__(self, patience: int, min_delta: float, restore_best_weights: bool):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

    def get_callbacks(self) -> List:
        from keras.callbacks import EarlyStopping

        return [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.patience,
                min_delta=self.min_delta,  # type: ignore
                restore_best_weights=self.restore_best_weights,
                verbose=0,
            )
        ]

    def should_use_validation_split(self) -> bool:
        return True


class NoOpEarlyStoppingStrategy(EarlyStoppingStrategy):
    """No early stopping (train for full epochs)."""

    def get_callbacks(self) -> List:
        return []

    def should_use_validation_split(self) -> bool:
        return False
