from __future__ import annotations

import gc
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import backend as K

from utils import logger
from utils.training import RunningConfusion, make_tf_dataset
from utils.loader import load_sparse_triplet_csv_gz

from models.factory import make_model

from context import TrainingContext, ModalityContext


# ------------------------------- Internal helpers -------------------------------


def _split_train_val(
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    val_fraction: float,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    n_train = int(x_train.shape[0])  # type: ignore[attr-defined]
    n_val = max(1, int(n_train * val_fraction))
    if n_val >= n_train:
        return x_train, y_train, x_train, y_train
    return x_train[:-n_val], y_train[:-n_val], x_train[-n_val:], y_train[-n_val:]  # type: ignore[index]


def _get_bottleneck_tensor(model: Model):
    """
    Return the output tensor of the required 'bottleneck' layer.
    Raises a clear error if the layer is missing.
    """
    try:
        return model.get_layer("bottleneck").output
    except ValueError as e:
        raise ValueError(
            "Required layer 'bottleneck' not found in model. "
            f"Available layers: {[layer.name for layer in model.layers]}"
        ) from e


def _extract_activations(
    model: Model, x: tf.Tensor, bottleneck_tensor, batch_size: int
) -> np.ndarray:
    """Build a light submodel to get bottleneck activations (batched)."""

    sub = Model(inputs=model.inputs, outputs=bottleneck_tensor)

    ds = (
        tf.data.Dataset.from_tensor_slices(x)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    acts = sub.predict(
        ds,
        verbose=0,  # type: ignore[arg-type]
    ).astype(np.float32)

    del sub
    K.clear_session()
    gc.collect()
    return acts


# ------------------------------------ Modality -----------------------------------


@dataclass
class ModalityResults:
    """Results from individual modality training."""

    modality_name: str
    activation_path: str
    accuracy: float
    tp: int
    fn: int
    fp: int
    tn: int
    features_per_individual: int


class Modality:
    """Encapsulates per-modality training logic and state."""

    def __init__(self, context: ModalityContext, training_context: TrainingContext):
        self.context = context
        self.training_context = training_context
        self.dataset: Optional[np.ndarray] = None

    # ------------------------------ Lifecycle -----------------------------------

    def initialize(self) -> None:
        """Initialize directories, logger, and load dataset."""
        self.context.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.context.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.init(str(self.context.log_file))

        self.dataset = load_sparse_triplet_csv_gz(str(self.context.dataset_path))

        logger.log(f"Initialized modality: {self.context.modality_name}")
        logger.log(f"Dataset shape: {self.dataset.shape}")

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.shutdown()
        K.clear_session()

    # ------------------------------ Training ------------------------------------

    def train(self) -> ModalityResults:
        """
        Execute CV training loop with configured strategies. Preserves outputs:
        - Saves per-test-index train/test bottlenecks for LOOCV
        - Saves fold-level bottlenecks for KFold
        - Returns ModalityResults with accuracy and confusion counts
        """
        assert self.dataset is not None, "Call initialize() before train()."

        labels = self.training_context.labels
        y_argmax = np.argmax(labels, axis=1)

        splitter = self._get_cv_splitter()
        is_kfold = self.training_context.cv_strategy == "kfold"
        n_folds = self.training_context.cv_folds if is_kfold else self.dataset.shape[0]
        log_interval = max(1, n_folds // 10)

        fold_train_acts: Optional[List[np.ndarray]] = [] if is_kfold else None
        fold_test_acts: Optional[List[np.ndarray]] = [] if is_kfold else None

        # Determine initial input size (before metadata, before PCA)
        base_input_size = self.dataset.shape[1]
        use_pca = self.context.preprocessing_strategy.should_apply(base_input_size)

        confusion = RunningConfusion()
        features_per_individual: Optional[int] = None

        for fold_idx, (tr_idx, te_idx) in enumerate(
            splitter.split(self.dataset, y_argmax)
        ):
            # ---- Split arrays for this fold (from original dataset)
            x_tr = self.dataset[tr_idx].copy()
            x_te = self.dataset[te_idx].copy()
            y_tr = labels[tr_idx]
            y_te = labels[te_idx]

            # ---- Per-fold metadata preparation (no data leakage)
            x_tr, x_te = self.context.metadata_strategy.prepare_fold(
                x_tr,
                x_te,
                self.training_context.metadata,
                tr_idx,
                te_idx,
            )

            fold_input_size = x_tr.shape[1]

            # ---- Optional PCA per fold
            if use_pca:
                input_size_before_pca = fold_input_size
                pca = self.context.preprocessing_strategy.fit(x_tr)
                x_tr = self.context.preprocessing_strategy.transform(x_tr, pca)
                x_te = self.context.preprocessing_strategy.transform(x_te, pca)
                fold_input_size = x_tr.shape[1]
                if fold_idx % log_interval == 0:
                    logger.log(
                        f"  Fold {fold_idx + 1}/{n_folds}: PCA {input_size_before_pca} -> {fold_input_size}"
                    )

            # ---- Optional pretraining (e.g., autoencoder) per fold
            pretrained_weights = None
            if self.context.pretraining_strategy.should_pretrain():
                if fold_idx % log_interval == 0:
                    logger.log(
                        f"  Fold {fold_idx + 1}/{n_folds}: Pretraining on {x_tr.shape[0]} samples..."
                    )
                pretrained_weights = self.context.pretraining_strategy.pretrain(
                    x_tr, fold_input_size
                )

            # ---- Build model for this fold
            model: Model = make_model(
                fold_input_size,
                "individual",
                pretrained_weights=pretrained_weights,
                optimizer_name=self.context.optimizer,
            )

            # Convert to tensors once
            x_tr_tf = tf.constant(x_tr, dtype=tf.float32)
            x_te_tf = tf.constant(x_te, dtype=tf.float32)
            y_tr_tf = tf.constant(y_tr, dtype=tf.float32)
            y_te_tf = tf.constant(y_te, dtype=tf.float32)

            # ---- Class weights
            class_weights = self.context.class_weighting_strategy.compute_weights(y_tr)

            # ---- (Optional) validation split + callbacks
            if self.context.early_stopping_strategy.should_use_validation_split():
                callbacks = self.context.early_stopping_strategy.get_callbacks()
                x_tr_sub, y_tr_sub, x_val, y_val = _split_train_val(
                    x_tr_tf,  # type: ignore[arg-type]
                    y_tr_tf,  # type: ignore[arg-type]
                    val_fraction=0.1,
                )
                train_ds = make_tf_dataset(
                    x_tr_sub, y_tr_sub, self.context.batch_size, shuffle=True
                )
                val_ds = make_tf_dataset(
                    x_val, y_val, self.context.batch_size, shuffle=False
                )
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=self.context.epochs,
                    verbose=0,  # type: ignore[arg-type]
                    class_weight=class_weights,
                    callbacks=callbacks,
                )
            else:
                train_ds = make_tf_dataset(
                    x_tr_tf,  # type: ignore[arg-type]
                    y_tr_tf,  # type: ignore[arg-type]
                    self.context.batch_size,
                    shuffle=True,
                )
                model.fit(
                    train_ds,
                    epochs=self.context.epochs,
                    verbose=0,  # type: ignore[arg-type]
                    class_weight=class_weights,
                )

            # ---- Predict on test fold
            test_ds = (
                tf.data.Dataset.from_tensor_slices(x_te_tf)
                .batch(self.context.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            preds = model.predict(
                test_ds,
                verbose=0,  # type: ignore[arg-type]
            )  # batched -> stable signature, no retrace spam

            # ---- Update confusion
            y_te_idx = np.argmax(y_te, axis=1)
            y_pr_idx = np.argmax(preds, axis=1)
            for t, p in zip(y_te_idx, y_pr_idx):
                confusion.update(t, p, pos_idx=0)

            # ---- Get the required 'bottleneck' tensor
            bottleneck_tensor = _get_bottleneck_tensor(model)

            # ---- Extract bottlenecks
            train_acts = _extract_activations(
                model,
                x_tr_tf,  # type: ignore[arg-type]
                bottleneck_tensor,
                self.context.batch_size,
            )
            test_acts = _extract_activations(
                model,
                x_te_tf,  # type: ignore[arg-type]
                bottleneck_tensor,
                self.context.batch_size,
            )

            # Validate dimension against expected bottleneck size (if configured)
            expected_dim = getattr(self.context, "bottleneck_dim", None)
            if expected_dim is not None and train_acts.shape[1] != expected_dim:
                raise ValueError(
                    f"Expected {expected_dim}-dim features, got {train_acts.shape[1]}-dim"
                )

            if features_per_individual is None:
                features_per_individual = int(train_acts.shape[1])

            # ---- Persist fold activations
            if is_kfold:
                fold_train_acts.append(train_acts)  # type: ignore[arg-type]
                fold_test_acts.append(test_acts)  # type: ignore[arg-type]
            else:
                # LOOCV: one held-out index; match your original filenames
                np.save(self.context.tmp_dir / f"train_{te_idx[0]}.npy", train_acts)
                np.save(self.context.tmp_dir / f"test_{te_idx[0]}.npy", test_acts)

            # ---- Periodic progress logs
            if confusion.total:
                msg = (
                    f"  [{self.context.modality_name}] Fold {fold_idx + 1}/{n_folds}: "
                    f"Running accuracy = {confusion.accuracy:.2%} "
                    f"({confusion.correct}/{confusion.total})"
                )
                print(msg, flush=True)
                if fold_idx % log_interval == 0 or fold_idx == n_folds - 1:
                    logger.log(msg)

            # ---- Free per-fold resources
            del model, x_tr_tf, x_te_tf, y_tr_tf, y_te_tf, preds, train_acts, test_acts
            K.clear_session()
            if fold_idx % 10 == 0:
                gc.collect()

        # ---- Save k-fold activation files, if applicable
        if is_kfold and fold_train_acts is not None and fold_test_acts is not None:
            logger.log(
                f"Saving {self.training_context.cv_folds} fold-level activation files..."
            )
            for fold_num in range(self.training_context.cv_folds):
                np.save(
                    self.context.tmp_dir / f"train_fold{fold_num}.npy",
                    fold_train_acts[fold_num],
                )
                np.save(
                    self.context.tmp_dir / f"test_fold{fold_num}.npy",
                    fold_test_acts[fold_num],
                )

        # ---- Final results
        return ModalityResults(
            modality_name=self.context.modality_name,
            activation_path=str(self.context.tmp_dir),
            accuracy=confusion.accuracy,
            tp=confusion.tp,
            fn=confusion.fn,
            fp=confusion.fp,
            tn=confusion.tn,
            features_per_individual=features_per_individual,  # type: ignore[arg-type]
        )

    # ------------------------------ CV splitter ----------------------------------

    def _get_cv_splitter(self):
        """Get appropriate CV splitter from training context."""
        if self.training_context.cv_strategy == "loocv":
            from sklearn.model_selection import LeaveOneOut

            return LeaveOneOut()
        else:
            from sklearn.model_selection import StratifiedKFold

            return StratifiedKFold(
                n_splits=self.training_context.cv_folds,
                shuffle=True,
                random_state=self.training_context.random_seed,
            )
