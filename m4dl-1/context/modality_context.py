from dataclasses import dataclass
from pathlib import Path
from strategies import (
    PreprocessingStrategy,
    PretrainingStrategy,
    MetadataStrategy,
    EarlyStoppingStrategy,
    ClassWeightingStrategy,
)


@dataclass
class ModalityContext:
    """Per-modality configuration and paths."""

    modality_name: str
    dataset_path: Path

    epochs: int
    batch_size: int
    optimizer: str

    bottleneck_dim: int

    tmp_dir: Path
    log_file: Path

    preprocessing_strategy: PreprocessingStrategy
    pretraining_strategy: PretrainingStrategy
    metadata_strategy: MetadataStrategy
    early_stopping_strategy: EarlyStoppingStrategy
    class_weighting_strategy: ClassWeightingStrategy
