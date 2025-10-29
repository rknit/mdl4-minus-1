from .base import ClassWeightingStrategy
import numpy as np


class BalancedWeightStrategy(ClassWeightingStrategy):
    """Compute balanced class weights using sklearn."""

    def compute_weights(self, y_train: np.ndarray) -> dict:
        from sklearn.utils.class_weight import compute_class_weight

        y_indices = np.argmax(y_train, axis=1)
        classes = np.unique(y_indices)
        weights = compute_class_weight('balanced', classes=classes, y=y_indices)
        return {i: weights[i] for i in range(len(classes))}


class NoOpWeightStrategy(ClassWeightingStrategy):
    """No class weighting (equal weights)."""

    def compute_weights(self, y_train: np.ndarray) -> None:
        return None
