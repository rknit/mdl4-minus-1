from __future__ import annotations

import gc
import os
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model

from utils import logger
from utils.training import RunningConfusion, make_tf_dataset
from models.factory import make_model

from context import TrainingContext, SharedModelContext
from models import ModalityResults

# ----------------------------------- SharedModel -----------------------------------


@dataclass
class SharedModelResults:
    """Results from shared model training."""

    accuracy: float
    tp: int
    fn: int
    fp: int
    tn: int


class SharedModel:
    """Encapsulates shared model training logic and state."""

    def __init__(self, context: SharedModelContext, training_context: TrainingContext):
        self.context = context
        self.training_context = training_context
        self.preloaded_data: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------ Lifecycle -----------------------------------

    def initialize(self) -> None:
        """Initialize directories and logger."""
        self.context.output_dir.mkdir(parents=True, exist_ok=True)
        logger.log("Initialized shared model")

    def cleanup(self) -> None:
        """Clean up resources."""
        K.clear_session()

    # --------------------------------- Public API --------------------------------

    def train(self, modality_results: Sequence[ModalityResults]) -> SharedModelResults:
        """Execute shared model training over concatenated modality bottlenecks."""
        individuals = [str(r.activation_path) for r in modality_results]
        self._validate_features(modality_results)

        is_kfold = self.training_context.cv_strategy == "kfold"
        n_folds = (
            self.training_context.cv_folds
            if is_kfold
            else len(self.training_context.labels)
        )

        # Load all activation files once (float32, shape-checked)
        self.preloaded_data = self._preload_activations(individuals, n_folds, is_kfold)

        features_per_modality = modality_results[0].features_per_individual
        input_size = features_per_modality * len(modality_results)

        # Build fresh base model + capture pristine weights
        base_model = make_model(
            input_size, "shared", optimizer_name=self.context.optimizer
        )
        initial_weights = base_model.get_weights()

        tp, fn, fp, tn, cnt = self._run_cv(
            preloaded_data=self.preloaded_data,
            individuals=individuals,
            ylab=self.training_context.labels,
            base_model=base_model,
            initial_weights=initial_weights,
            input_size=input_size,
            n_folds=n_folds,
            is_kfold=is_kfold,
        )

        total = tp + tn + fp + fn
        accuracy = cnt / total if total > 0 else 0.0

        return SharedModelResults(accuracy=accuracy, tp=tp, fn=fn, fp=fp, tn=tn)

    def save_results(
        self,
        individual_results: Sequence[ModalityResults],
        shared_results: SharedModelResults,
    ) -> None:
        """Save all results to files."""
        from utils.reporter import save_results_to_files, print_summary

        individual_list = [
            {
                "name": r.modality_name,
                "accuracy": r.accuracy,
                "TP": r.tp,
                "FN": r.fn,
                "FP": r.fp,
                "TN": r.tn,
                "features_per_individual": r.features_per_individual,
            }
            for r in individual_results
        ]
        individual_dict = {
            r.modality_name: individual_list[i]
            for i, r in enumerate(individual_results)
        }
        _ = individual_dict  # kept for compatibility; not used directly below

        shared_dict = {
            "name": "Shared",
            "accuracy": shared_results.accuracy,
            "TP": shared_results.tp,
            "FN": shared_results.fn,
            "FP": shared_results.fp,
            "TN": shared_results.tn,
        }

        print_summary(individual_list, shared_dict)
        save_results_to_files(
            individual_list,
            shared_dict,
            self.training_context.output_base_dir,
            self.training_context.disease_type,
            self.training_context.timestamp,
        )

    # ------------------------------ Internal: IO & checks ------------------------

    def _preload_activations(
        self, individuals: List[str], n_folds: int, is_kfold: bool
    ) -> Dict[str, np.ndarray]:
        """Preload all activation NPY files into memory (float32, 2D) with strict shape checks."""
        preloaded_data: Dict[str, np.ndarray] = {}

        total_files = len(individuals) * n_folds * 2  # train + test per fold
        loaded_count = 0
        log_interval = max(1, total_files // 10)

        expected_features: Optional[int] = None

        for indi in individuals:
            for i in range(n_folds):
                if is_kfold:
                    train_key = f"{indi}_train_fold{i}"
                    test_key = f"{indi}_test_fold{i}"
                    train_file_npy = os.path.join(indi, f"train_fold{i}.npy")
                    test_file_npy = os.path.join(indi, f"test_fold{i}.npy")
                else:
                    train_key = f"{indi}_train_{i}"
                    test_key = f"{indi}_test_{i}"
                    train_file_npy = os.path.join(indi, f"train_{i}.npy")
                    test_file_npy = os.path.join(indi, f"test_{i}.npy")

                # --- Train
                if not os.path.exists(train_file_npy):
                    raise FileNotFoundError(
                        f"Missing activation file: {train_file_npy}"
                    )
                train_data = np.load(train_file_npy).astype(np.float32)
                if train_data.ndim != 2:
                    raise ValueError(
                        f"Invalid train activation shape in {train_file_npy}: expected 2D, got {train_data.ndim}D"
                    )

                if expected_features is None:
                    expected_features = int(train_data.shape[1])
                elif train_data.shape[1] != expected_features:
                    raise ValueError(
                        f"Feature count mismatch in {train_file_npy}: expected {expected_features}, got {train_data.shape[1]}"
                    )
                preloaded_data[train_key] = train_data

                loaded_count += 1
                if loaded_count % log_interval == 0 or loaded_count == total_files:
                    pct = (loaded_count / total_files) * 100
                    logger.log(
                        f"Loading activation files: {loaded_count}/{total_files} ({pct:.0f}%)"
                    )

                # --- Test
                if not os.path.exists(test_file_npy):
                    raise FileNotFoundError(f"Missing activation file: {test_file_npy}")
                test_data = np.load(test_file_npy).astype(np.float32)
                if test_data.ndim != 2:
                    raise ValueError(
                        f"Invalid test activation shape in {test_file_npy}: expected 2D, got {test_data.ndim}D"
                    )
                if test_data.shape[1] != expected_features:
                    raise ValueError(
                        f"Feature count mismatch in {test_file_npy}: expected {expected_features}, got {test_data.shape[1]}"
                    )
                preloaded_data[test_key] = test_data

                loaded_count += 1
                if loaded_count % log_interval == 0 or loaded_count == total_files:
                    pct = (loaded_count / total_files) * 100
                    logger.log(
                        f"Loading activation files: {loaded_count}/{total_files} ({pct:.0f}%)"
                    )

        logger.log(
            f"Successfully loaded {len(preloaded_data)} activation files. "
            f"Feature dimension per modality: {expected_features}"
        )
        return preloaded_data

    def _validate_features(self, modality_results) -> None:
        """Validate all modalities have consistent feature count."""
        features = [r.features_per_individual for r in modality_results]
        if len(set(features)) != 1:
            raise ValueError(
                f"Inconsistent feature counts across modalities: {features}"
            )

    # ------------------------------ Internal: CV loop ----------------------------

    def _run_cv(
        self,
        preloaded_data: Dict[str, np.ndarray],
        individuals: List[str],
        ylab: np.ndarray,
        base_model: Model,
        initial_weights: List[np.ndarray],
        input_size: int,
        n_folds: int,
        is_kfold: bool,
    ) -> Tuple[int, int, int, int, int]:
        """Execute cross-validation folds sequentially."""
        if is_kfold:
            from sklearn.model_selection import StratifiedKFold

            y_class = np.argmax(ylab, axis=1)
            kfold = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.training_context.random_seed,
            )
            fold_splits = list(kfold.split(np.zeros((len(y_class), 1)), y_class))

        log_interval = max(1, n_folds // 10)
        confusion = RunningConfusion()

        for fold_idx in range(n_folds):
            # ---- Assemble concatenated train/test sets
            if is_kfold:
                train_blocks = [
                    preloaded_data[f"{indi}_train_fold{fold_idx}"]
                    for indi in individuals
                ]
                test_blocks = [
                    preloaded_data[f"{indi}_test_fold{fold_idx}"]
                    for indi in individuals
                ]
                train_idx, test_idx = fold_splits[fold_idx]
                train_y = ylab[train_idx]
                test_y = ylab[test_idx]
            else:
                train_blocks = [
                    preloaded_data[f"{indi}_train_{fold_idx}"] for indi in individuals
                ]
                test_blocks = [
                    preloaded_data[f"{indi}_test_{fold_idx}"] for indi in individuals
                ]
                train_y = np.delete(ylab, fold_idx, axis=0)
                test_y = ylab[fold_idx : fold_idx + 1]

            train_set = np.concatenate(train_blocks, axis=1).astype(
                np.float32, copy=False
            )
            test_set = np.concatenate(test_blocks, axis=1).astype(
                np.float32, copy=False
            )

            train_y = np.asarray(train_y, dtype=np.float32)
            test_y = np.asarray(test_y, dtype=np.float32)

            # ---- Reset weights/optimizer per fold
            base_model.set_weights(initial_weights)

            # ---- Train (unchanged)
            class_weight_dict = self.context.class_weighting_strategy.compute_weights(
                train_y
            )
            if self.context.early_stopping_strategy.should_use_validation_split():
                callbacks = self.context.early_stopping_strategy.get_callbacks()
                n_train = len(train_set)
                n_val = max(1, int(n_train * 0.1))
                if n_val < n_train:
                    train_set_sub, train_y_sub = train_set[:-n_val], train_y[:-n_val]
                    val_set, val_y = train_set[-n_val:], train_y[-n_val:]
                else:
                    train_set_sub, train_y_sub = train_set, train_y
                    val_set, val_y = train_set, train_y

                train_ds = make_tf_dataset(
                    train_set_sub, train_y_sub, self.context.batch_size, shuffle=True
                )
                val_ds = make_tf_dataset(
                    val_set, val_y, self.context.batch_size, shuffle=False
                )
                base_model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.context.epochs,
                    verbose=0,  # type: ignore[arg-type]
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                )
            else:
                train_ds = make_tf_dataset(
                    train_set, train_y, self.context.batch_size, shuffle=True
                )
                base_model.fit(
                    train_ds,
                    epochs=self.context.epochs,
                    verbose=0,  # type: ignore[arg-type]
                    class_weight=class_weight_dict,
                )

            # ---- Predict (BATCHED to avoid retracing)
            test_ds = (
                tf.data.Dataset.from_tensor_slices(test_set)
                .batch(self.context.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            predictions = base_model.predict(
                test_ds,
                verbose=0,  # type: ignore[arg-type]
            )

            # ---- Metrics
            y_true = np.argmax(test_y, axis=1)
            y_pred = np.argmax(predictions, axis=1)
            for t, p in zip(y_true, y_pred):
                confusion.update(t, p, pos_idx=0)

            # ---- Progress logs
            if confusion.total:
                msg = (
                    f"  Fold {fold_idx + 1}/{n_folds}: "
                    f"Running accuracy = {confusion.accuracy:.2%} "
                    f"({confusion.correct}/{confusion.total})"
                )
                print(msg, flush=True)
                if fold_idx % log_interval == 0 or fold_idx == n_folds - 1:
                    logger.log(msg)

            # ---- Fresh model for next fold (prevents graph growth)
            if fold_idx < n_folds - 1:
                K.clear_session()
                base_model = make_model(
                    input_size, "shared", optimizer_name=self.context.optimizer
                )
                initial_weights = base_model.get_weights()

            if fold_idx % 10 == 0:
                gc.collect()

        return confusion.tp, confusion.fn, confusion.fp, confusion.tn, confusion.correct
