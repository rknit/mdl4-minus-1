"""GPU and XLA utilities for dynamic VRAM allocation, monitoring, and safety checks."""

import tensorflow as tf
from . import logger
import subprocess


# =========================================
# XLA Support Check
# =========================================


def ensure_xla_supported_or_disable(silent: bool = False) -> bool:
    """Check whether XLA is available and functional, and disable it if not.

    Returns:
        bool: True if XLA is available and enabled, False otherwise.
    """
    try:
        # Detect if any XLA device is present
        xla_devices = [
            d.name for d in tf.config.list_logical_devices() if "XLA" in d.name
        ]
        if not xla_devices:
            tf.config.optimizer.set_jit(False)
            if not silent:
                logger.log("XLA not available — JIT disabled.")
            return False

        # Try a trivial JIT compile to ensure functional XLA
        @tf.function(jit_compile=True)
        def _probe():
            return tf.constant(1.0) + tf.constant(1.0)  # type: ignore[return]

        _ = _probe()  # type: ignore[call-arg]
        if not silent:
            logger.log("XLA available and functional.")
        return True

    except Exception as e:
        # Disable XLA globally as a safety fallback
        try:
            tf.config.optimizer.set_jit(False)
        except Exception:
            pass
        if not silent:
            logger.log(f"XLA probe failed — disabled for safety. Reason: {e}")
        return False


# =========================================
# GPU Memory Utilities
# =========================================


def get_gpu_memory_info():
    """Get total GPU memory in MB.

    Returns:
        int: Total GPU memory in MB, or 0 if no GPU available.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return 0

        # Use nvidia-smi for accurate VRAM detection
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # Fallback conservative estimate (assume 4GB)
        return 4096

    except Exception:
        return 0


def calculate_memory_per_worker(
    total_vram_mb, n_workers, reserve_mb=512, safety_factor=0.9
):
    """Calculate safe GPU memory limit per worker for parallel execution."""
    if total_vram_mb <= 0:
        return 0

    available_mb = total_vram_mb - reserve_mb
    if available_mb <= 0:
        return 0

    per_worker_mb = int((available_mb / n_workers) * safety_factor)

    logger.log(
        f"GPU: {total_vram_mb}MB total, {per_worker_mb}MB per worker ({n_workers} workers)"
    )

    if per_worker_mb < 800:
        logger.log(
            f"⚠️ WARNING: Low GPU memory per worker ({per_worker_mb}MB) — consider reducing batch size"
        )

    return per_worker_mb


def check_gpu_viability_for_parallel(total_vram_mb, n_workers, min_per_worker_mb=800):
    """Check if GPU has sufficient memory for parallel execution."""
    per_worker_mb = calculate_memory_per_worker(total_vram_mb, n_workers)

    if per_worker_mb < min_per_worker_mb:
        message = (
            f"Insufficient GPU memory for {n_workers} parallel workers.\n"
            f"Available: ~{per_worker_mb}MB per worker (Required: {min_per_worker_mb}MB)\n"
            f"Recommendations:\n"
            f"  1. Use CPU mode with parallel modalities (-pm, no --use-gpu)\n"
            f"  2. Use GPU mode without parallel modalities (--use-gpu, no -pm)\n"
            f"  3. Reduce batch size (e.g., -b1 16 or -b1 8)"
        )
        return False, per_worker_mb, message

    message = f"GPU memory sufficient for {n_workers} parallel workers (~{per_worker_mb}MB each)"
    return True, per_worker_mb, message


def configure_gpu_memory(gpu_memory_limit_mb=None):
    """Configure GPU memory settings (growth and optional limit)."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return False

        for gpu in gpus:
            # Enable dynamic growth to prevent full preallocation
            tf.config.experimental.set_memory_growth(gpu, True)

            # Optionally limit memory per GPU (useful for parallel workers)
            if gpu_memory_limit_mb is not None:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_memory_limit_mb
                        )
                    ],
                )

        return True

    except RuntimeError as e:
        logger.log(f"ERROR: Could not configure GPU memory: {e}")
        return False
    except Exception as e:
        logger.log(f"ERROR: Unexpected GPU error: {e}")
        return False
