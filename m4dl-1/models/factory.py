from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, SGD
import numpy as np

from utils import logger


# ----------------------------- internal helpers -----------------------------


def _select_optimizer(name: str):
    """Return a configured optimizer."""
    name = (name or "adam").lower()
    if name == "sgd":
        # Nesterov helps small, sparse-ish datasets; lr kept conservative for LOOCV stability
        return SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    elif name == "adam":
        return Adam(learning_rate=0.001)
    else:
        logger.log(
            f"[make_model] Unrecognized optimizer '{name}', falling back to Adam(1e-3)."
        )
        return Adam(learning_rate=0.001)


def _set_layer_weights(layer: Dense, weights: Sequence[np.ndarray]) -> None:
    """Set [kernel, bias] on a Dense layer with shape checks."""
    if len(weights) != 2:
        raise ValueError(
            f"Expected 2 arrays [kernel, bias] for layer '{layer.name}', got {len(weights)}."
        )
    kernel, bias = weights

    # Ensure variables exist. If the layer isn't built yet, build it using the kernel's in_dim.
    if not getattr(layer, "built", False):
        if kernel.ndim != 2:
            raise ValueError(
                f"Cannot build layer '{layer.name}': kernel must be 2D, got {kernel.shape}."
            )
        in_dim = int(kernel.shape[0])
        layer.build((None, in_dim))

    # Use actual variable shapes—robust across Keras versions.
    expected_kernel_shape = tuple(
        int(d) for d in layer.kernel.shape
    )  # (in_dim, out_dim)

    # (out_dim,)
    expected_bias_shape = tuple(int(d) for d in layer.bias.shape)  # type: ignore

    if kernel.shape != expected_kernel_shape:
        raise ValueError(
            f"Kernel shape mismatch for '{layer.name}': expected {expected_kernel_shape}, got {kernel.shape}."
        )
    if bias.shape != expected_bias_shape:
        raise ValueError(
            f"Bias shape mismatch for '{layer.name}': expected {expected_bias_shape}, got {bias.shape}."
        )

    layer.set_weights([kernel, bias])


def _load_pretrained_encoder_weights(
    model: Model,
    pretrained_weights: Union[Sequence[np.ndarray], Dict[str, Sequence[np.ndarray]]],
    encoder_layer_names: Sequence[str],
) -> None:
    """
    Load weights into the encoder stack (e.g., dense1, dense2, bottleneck).
    Supports:
      - list/tuple: [k1, b1, k2, b2, k3, b3]
      - dict: {"dense1": [k,b], "dense2": [k,b], "bottleneck": [k,b]}
    """
    if isinstance(pretrained_weights, dict):
        for lname in encoder_layer_names:
            if lname not in pretrained_weights:
                logger.log(
                    f"[make_model] Pretrained dict missing layer '{lname}'; skipping it."
                )
                continue
            layer = model.get_layer(lname)
            _set_layer_weights(layer, pretrained_weights[lname])
        logger.log("[make_model] Loaded pretrained encoder weights from dict.")
        return

    # Sequence case: flat kernel/bias pairs
    if not isinstance(pretrained_weights, (list, tuple)):
        raise TypeError(
            "pretrained_weights must be None, a list/tuple of arrays, or a dict of layer_name -> [kernel, bias]."
        )

    expected_len = 2 * len(encoder_layer_names)
    if len(pretrained_weights) < expected_len:
        raise ValueError(
            f"Not enough arrays in pretrained_weights: expected at least {expected_len} "
            f"([k,b] per encoder layer), got {len(pretrained_weights)}."
        )

    w_iter = iter(pretrained_weights)
    for lname in encoder_layer_names:
        layer = model.get_layer(lname)
        kernel = next(w_iter)
        bias = next(w_iter)
        _set_layer_weights(layer, [kernel, bias])

    logger.log("[make_model] Loaded pretrained encoder weights from flat list/tuple.")


# --------------------------------- main API ---------------------------------


def make_model(
    input_size: int,
    model_type: str,
    pretrained_weights: Optional[
        Union[Sequence[np.ndarray], Dict[str, Sequence[np.ndarray]]]
    ] = None,
    optimizer_name: str = "adam",
) -> Sequential:
    """
    Create and compile a model.

    - Architecture mirrors original MDL4Microbiome baseline (no BN/Dropout).
    - Adds a named bottleneck layer 'bottleneck' so downstream code can resolve by name.

    Args:
        input_size: Input dimension.
        model_type: 'individual' or 'shared'.
        pretrained_weights: Optional encoder weights:
            * list/tuple: [k1, b1, k2, b2, k3, b3]
            * dict: { 'dense1': [k,b], 'dense2': [k,b], 'bottleneck': [k,b] }
        optimizer_name: 'adam' (default) or 'sgd'.

    Returns:
        Compiled keras Sequential model.
    """
    model_type = model_type.lower().strip()

    if model_type == "individual":
        # Encoder: dense1 -> dense2 -> bottleneck
        model = Sequential(
            [
                Input(shape=(input_size,), name="input"),
                Dense(200, activation="relu", name="dense1"),
                Dense(100, activation="relu", name="dense2"),
                Dense(50, activation="relu", name="bottleneck"),  # Named bottleneck
                Dense(2, activation="softmax", name="output"),
            ]
        )
        # Load pretrained weights for encoder if provided
        if pretrained_weights is not None:
            model.build(
                (None, input_size)
            )  # Ensure model is built before setting weights
            _load_pretrained_encoder_weights(
                model,
                pretrained_weights,
                encoder_layer_names=("dense1", "dense2", "bottleneck"),
            )

    elif model_type == "shared":
        # Keep a small shared head; still expose a named bottleneck for consistency
        model = Sequential(
            [
                Input(shape=(input_size,), name="input"),
                Dense(50, activation="relu", name="dense1_shared"),
                Dense(25, activation="relu", name="bottleneck"),  # Named bottleneck
                Dense(2, activation="softmax", name="output"),
            ]
        )
        # Intentionally ignore pretrained weights for 'shared' (not encoder-based)
        if pretrained_weights is not None:
            logger.log(
                "[make_model] Ignoring pretrained_weights for model_type='shared'."
            )

    else:
        raise ValueError(
            "make_model(): unexpected model_type (expected 'individual' or 'shared')."
        )

    optimizer = _select_optimizer(optimizer_name)

    # XLA JIT is managed globally; keep compile simple and LOOCV-friendly.
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,  # type: ignore
        metrics=["accuracy"],
    )
    return model
