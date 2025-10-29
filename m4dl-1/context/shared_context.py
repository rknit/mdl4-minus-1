from dataclasses import dataclass
from pathlib import Path
from typing import List
from strategies import EarlyStoppingStrategy, ClassWeightingStrategy


@dataclass
class SharedModelContext:
    """Shared model configuration and paths."""

    modality_activation_dirs: List[Path]
    features_per_modality: int

    epochs: int
    batch_size: int
    optimizer: str

    output_dir: Path
    log_file: Path

    early_stopping_strategy: EarlyStoppingStrategy
    class_weighting_strategy: ClassWeightingStrategy
