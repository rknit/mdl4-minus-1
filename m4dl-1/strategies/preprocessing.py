from .base import PreprocessingStrategy
import numpy as np
from typing import Any


class PCAStrategy(PreprocessingStrategy):
    """Per-fold PCA with configurable variance threshold."""

    def __init__(self, variance_threshold: float = 0.95, min_features: int = 1000):
        self.variance_threshold = variance_threshold
        self.min_features = min_features

    def should_apply(self, n_features: int) -> bool:
        return n_features > self.min_features

    def fit(self, X_train: np.ndarray) -> Any:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.variance_threshold, svd_solver='full')
        pca.fit(X_train)
        return pca

    def transform(self, X: np.ndarray, transformer: Any) -> np.ndarray:
        return transformer.transform(X).astype('float32')


class NoOpPreprocessingStrategy(PreprocessingStrategy):
    """Pass-through preprocessing (no PCA)."""

    def should_apply(self, n_features: int) -> bool:
        return False

    def fit(self, X_train: np.ndarray) -> None:
        return None

    def transform(self, X: np.ndarray, transformer: Any) -> np.ndarray:
        return X
