from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np
import pandas as pd


class PreprocessingStrategy(ABC):
    """Abstract base for preprocessing (e.g., PCA)."""

    @abstractmethod
    def should_apply(self, n_features: int) -> bool:
        """Check if preprocessing should be applied."""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> Any:
        """Fit on training data, return fitted transformer."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, transformer: Any) -> np.ndarray:
        """Transform data using fitted transformer."""
        pass


class PretrainingStrategy(ABC):
    """Abstract base for pretraining (e.g., autoencoder)."""

    @abstractmethod
    def should_pretrain(self) -> bool:
        """Check if pretraining is enabled."""
        pass

    @abstractmethod
    def pretrain(
        self, X_train: np.ndarray, n_input_features: int
    ) -> Optional[List[np.ndarray]]:
        """Pretrain and return encoder weights, or None."""
        pass


class MetadataStrategy(ABC):
    """Abstract base for metadata handling."""

    @abstractmethod
    def should_use_metadata(self) -> bool:
        """Check if metadata should be used."""
        pass

    @abstractmethod
    def prepare_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        metadata: Optional[pd.DataFrame],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare train and test sets with metadata for a single CV fold.

        Fits preprocessing (imputation, scaling) on training data only to avoid data leakage.
        Then transforms both train and test sets using the fitted preprocessors.

        Args:
            X_train: Training features for this fold
            X_test: Test features for this fold
            metadata: Full metadata DataFrame
            train_idx: Indices of training samples in original dataset
            test_idx: Indices of test samples in original dataset

        Returns:
            Tuple of (X_train_prepared, X_test_prepared)
        """
        pass


class EarlyStoppingStrategy(ABC):
    """Abstract base for early stopping."""

    @abstractmethod
    def get_callbacks(self) -> List:
        """Return list of Keras callbacks (empty if disabled)."""
        pass

    @abstractmethod
    def should_use_validation_split(self) -> bool:
        """Check if validation split should be used."""
        pass


class ClassWeightingStrategy(ABC):
    """Abstract base for class weighting."""

    @abstractmethod
    def compute_weights(self, y_train: np.ndarray) -> Optional[dict]:
        """Compute class weights, or None if disabled."""
        pass
