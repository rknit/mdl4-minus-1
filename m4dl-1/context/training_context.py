from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import os
from utils.config import StructuredConfig
from utils import logger
from utils.platform import (
    ensure_xla_supported_or_disable,
    configure_gpu_memory,
    get_gpu_memory_info,
    check_gpu_viability_for_parallel,
)


@dataclass
class TrainingContext:
    """Global configuration and shared state for entire training run."""

    config: StructuredConfig
    disease_type: str
    timestamp: str
    labels: np.ndarray
    metadata: Optional[pd.DataFrame]
    cv_strategy: str
    cv_folds: int
    random_seed: int
    use_gpu: bool
    parallel_modalities: bool

    data_dir: Path
    logs_base_dir: Path
    tmp_base_dir: Path
    output_base_dir: Path
    datapaths: List[str] = field(default_factory=list)

    def initialize(self):
        """Setup TensorFlow, GPU, base directories, main logger."""
        import tensorflow as tf

        self.logs_base_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_base_dir / "main.log"
        logger.init(str(log_file))

        logger.log(f"Starting training for {self.disease_type}")
        logger.log(f"CV strategy: {self.cv_strategy}")
        logger.log(f"GPU enabled: {self.use_gpu}")
        logger.log(f"Parallel modalities: {self.parallel_modalities}")
        logger.log("Resolved directory paths:")
        logger.log(f"  data_dir: {self.data_dir}")
        logger.log(f"  logs_base_dir: {self.logs_base_dir}")
        logger.log(f"  tmp_base_dir: {self.tmp_base_dir}")
        logger.log(f"  output_base_dir: {self.output_base_dir}")

        tf.get_logger().setLevel("ERROR")

        if self.use_gpu:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                logger.log(f"Found {len(gpus)} GPU(s)")
                gpu_configured = configure_gpu_memory()
                if not gpu_configured:
                    logger.log("Failed to configure GPU memory, falling back to CPU")
                    self.use_gpu = False

                # Check GPU viability for parallel execution
                if self.use_gpu and self.parallel_modalities:
                    total_vram_mb = get_gpu_memory_info()
                    n_workers = len(self.datapaths)
                    is_viable, per_worker_mb, message = (
                        check_gpu_viability_for_parallel(total_vram_mb, n_workers)
                    )

                    if is_viable:
                        logger.log(message)
                    else:
                        logger.log(f"⚠️ {message}")
                        # Don't fail, just warn - user may have reduced batch sizes
            else:
                logger.log("GPU requested but not found, falling back to CPU")
                self.use_gpu = False

        # Configure XLA JIT compilation
        xla_enabled = ensure_xla_supported_or_disable(silent=False)
        if xla_enabled and self.use_gpu:
            gpu_devices = tf.config.list_physical_devices("GPU")
            if gpu_devices:
                os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"

    def cleanup(self):
        """Shutdown main logger, final cleanup."""
        logger.log("Training completed")
        logger.shutdown()
