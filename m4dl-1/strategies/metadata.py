from .base import MetadataStrategy
import numpy as np
import pandas as pd
from typing import Optional


class ConcatenateMetadataStrategy(MetadataStrategy):
    """Normalize and concatenate metadata as additional features."""

    def should_use_metadata(self) -> bool:
        return True

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
        Fits imputer and scaler on training data only (no data leakage).
        """
        if metadata is None or metadata.empty:
            return X_train, X_test

        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # Extract metadata for train and test
        metadata_train = metadata.iloc[train_idx].values.astype(np.float32)
        metadata_test = metadata.iloc[test_idx].values.astype(np.float32)

        # Fit imputer on TRAINING data only
        imputer = SimpleImputer(strategy="mean")
        metadata_train_imputed = imputer.fit_transform(metadata_train)
        metadata_test_imputed = imputer.transform(
            metadata_test
        )  # Transform only, no fit

        # Fit scaler on TRAINING data only
        scaler = StandardScaler()
        metadata_train_normalized = scaler.fit_transform(metadata_train_imputed)
        metadata_test_normalized = scaler.transform(
            metadata_test_imputed
        )  # Transform only, no fit

        # Concatenate with original features
        X_train_combined = np.concatenate(
            [X_train, metadata_train_normalized], axis=1
        ).astype("float32")
        X_test_combined = np.concatenate(
            [X_test, metadata_test_normalized], axis=1
        ).astype("float32")

        return X_train_combined, X_test_combined


class NoOpMetadataStrategy(MetadataStrategy):
    """Do not use metadata."""

    def should_use_metadata(self) -> bool:
        return False

    def prepare_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        metadata: Optional[pd.DataFrame],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """No-op: return data unchanged."""
        return X_train, X_test
