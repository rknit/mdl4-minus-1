from __future__ import annotations

from typing import Dict, List
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import backend as K

from .base import PretrainingStrategy


class AutoencoderStrategy(PretrainingStrategy):
    """Autoencoder-based unsupervised pretraining."""

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        optimizer: str,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.optimizer = str(optimizer).lower()

    def should_pretrain(self) -> bool:
        return True

    def _make_optimizer(self):
        from keras.optimizers import Adam, SGD

        if self.optimizer == "sgd":
            return SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
        return Adam(learning_rate=0.001)

    def _build_autoencoder(self, n_input_features: int) -> Model:
        """Encoder: 200 → 100 → 50 (bottleneck-compatible). Decoder mirrors."""
        encoder = Sequential(
            [
                Input(shape=(n_input_features,), name="pretrain_input"),
                Dense(200, activation="relu", name="pretrain_dense1"),
                Dense(100, activation="relu", name="pretrain_dense2"),
                Dense(50, activation="relu", name="pretrain_dense3"),
            ],
            name="encoder",
        )
        decoder = Sequential(
            [
                Dense(100, activation="relu", name="decoder_dense1"),
                Dense(200, activation="relu", name="decoder_dense2"),
                Dense(n_input_features, activation="linear", name="decoder_output"),
            ],
            name="decoder",
        )
        return Sequential([encoder, decoder], name="autoencoder")

    def _make_ds(
        self,
        X: np.ndarray,
        *,
        shuffle: bool,
        batch_size: int,
    ) -> tf.data.Dataset:
        """Build a tf.data pipeline X→X. Pylance-safe (no shape-index surprises)."""
        X_tf = tf.convert_to_tensor(X)
        ds = tf.data.Dataset.from_tensor_slices((X_tf, X_tf))
        if shuffle:
            # full-dataset shuffle is fine for small tabular data
            n = int(X.shape[0])
            buf = n if n > 0 else 1
            ds = ds.shuffle(buffer_size=buf, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def pretrain(
        self, X_train: np.ndarray, n_input_features: int
    ) -> Dict[str, List[np.ndarray]]:
        """
        Pretrain autoencoder and return encoder weights mapped to downstream layer names:
          { "dense1": [kernel, bias], "dense2": [kernel, bias], "bottleneck": [kernel, bias] }
        """

        if X_train.ndim != 2 or X_train.shape[1] != n_input_features:
            raise ValueError(
                f"X_train must be 2D with shape (_, {n_input_features}), got {X_train.shape}"
            )

        X_train = X_train.astype(np.float32, copy=False)

        model = self._build_autoencoder(n_input_features)
        model.compile(
            optimizer=self._make_optimizer(),  # type: ignore[arg-type]
            loss="mse",
            metrics=["mae"],
        )

        # Dataset path (no validation)
        ds = self._make_ds(X_train, shuffle=True, batch_size=self.batch_size)
        model.fit(
            ds,
            epochs=self.epochs,
            verbose=0,  # type: ignore[arg-type]
        )

        # --- Extract encoder weights and map to downstream expected names ---
        # Your downstream model expects names: "dense1", "dense2", "bottleneck"
        # The pretrain encoder used: "pretrain_dense1/2/3"
        encoder: Sequential = model.layers[0]  # type: ignore[assignment]
        p_dense1 = encoder.get_layer("pretrain_dense1")
        p_dense2 = encoder.get_layer("pretrain_dense2")
        p_dense3 = encoder.get_layer("pretrain_dense3")

        w_dense1 = p_dense1.get_weights()  # [kernel, bias]
        w_dense2 = p_dense2.get_weights()
        w_dense3 = p_dense3.get_weights()

        # Basic shape sanity (helpful after PCA)
        def _expect_shapes(w, in_dim: int, out_dim: int, name: str):
            k, b = w
            if k.shape != (in_dim, out_dim) or b.shape != (out_dim,):
                raise ValueError(
                    f"Pretrained layer '{name}' weights have wrong shape: "
                    f"kernel {k.shape} vs {(in_dim, out_dim)}, bias {b.shape} vs {(out_dim,)}"
                )

        _expect_shapes(w_dense1, n_input_features, 200, "pretrain_dense1")
        _expect_shapes(w_dense2, 200, 100, "pretrain_dense2")
        _expect_shapes(w_dense3, 100, 50, "pretrain_dense3")

        pretrained: Dict[str, List[np.ndarray]] = {
            "dense1": [w_dense1[0], w_dense1[1]],
            "dense2": [w_dense2[0], w_dense2[1]],
            "bottleneck": [w_dense3[0], w_dense3[1]],
        }

        # Cleanup to keep GPU/CPU memory stable across folds
        K.clear_session()
        del model

        return pretrained


class NoOpPretrainingStrategy(PretrainingStrategy):
    """No pretraining (train from random initialization)."""

    def should_pretrain(self) -> bool:
        return False

    def pretrain(self, X_train: np.ndarray, n_input_features: int) -> None:
        return None
