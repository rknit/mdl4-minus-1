from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tomli as tomllib

# ============================== Template ===============================

def generate_config_template() -> str:
    """
    Generate a concise TOML configuration template (baseline settings).

    Returns:
        str: Complete TOML template as a string (with trailing newline)
    """
    return """# MDL4Microbiome(-1) Configuration (Baseline - exact replication of original MDL4)

[experiment]
# Valid: "T2D", "CRC", "IBD", "cirrhosis"
disease_type = "T2D"

[training]
indi_epochs = 30
shared_epochs = 20
indi_batch = 1
shared_batch = 1
optimizer = "adam"  # "adam" | "sgd"

[pretraining]
enabled = false
epochs = 50

[early_stopping]
enabled = false
patience = 10
min_delta = 0.0001
restore_best_weights = true

[cross_validation]
# "loocv" or "kfold"
strategy = "loocv"
# Used only when strategy = "kfold"
folds = 10
# Random seed for k-fold reproducibility (LOOCV is deterministic)
random_seed = 42

[performance]
parallel_modalities = false
use_gpu = false

[preprocessing]
pca_enabled = false
pca_variance_threshold = 0.95
pca_min_features = 1000

[data]
use_metadata = false
class_weighting = false

[paths]
# Base directories (supports {disease_type} and {timestamp} placeholders)
data_dir = "./data/{disease_type}"
logs_base_dir = "./logs/{disease_type}/{timestamp}"
tmp_base_dir = "./tmp/{disease_type}"
output_base_dir = "./output/{disease_type}/{timestamp}"
"""

def write_template_to_file(output_path: str) -> bool:
    """
    Write the configuration template to a file.

    Args:
        output_path: Path where the template should be written

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generate_config_template())
        return True
    except Exception as e:  # pragma: no cover
        print(f"Error writing config template to {output_path}: {e}")
        return False


# ============================== Errors ================================

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


# ========================== Raw Validated Config ======================

_ALLOWED_DISEASES = {"T2D", "CRC", "IBD", "cirrhosis"}
_ALLOWED_OPTIMIZERS = {"adam", "sgd"}
_REQUIRED_SECTIONS = {"experiment", "training", "cross_validation", "performance"}
_OPTIONAL_SECTIONS = {"early_stopping", "preprocessing", "pretraining", "data", "paths"}
_KNOWN_SECTIONS = _REQUIRED_SECTIONS | _OPTIONAL_SECTIONS


class Config:
    """
    Loader + validator/coercion for TOML config.
    Keep this as the single source of truth; adapt to StructuredConfig for use.
    """

    # public attrs populated during validation (kept for mypy/readability)
    disease_type: str
    indi_epochs: int
    shared_epochs: int
    indi_batch: int
    shared_batch: int
    optimizer: str
    pretrain_enabled: bool
    pretrain_epochs: int
    cv_folds: Optional[int]
    random_seed: int
    parallel_modalities: bool
    use_gpu: bool
    early_stopping_enabled: bool
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_restore_best_weights: bool
    pca_enabled: bool
    pca_variance_threshold: float
    pca_min_features: int
    use_metadata: bool
    class_weighting: bool
    data_dir: str
    logs_base_dir: str
    tmp_base_dir: str
    output_base_dir: str

    def __init__(self, config_dict: Dict[str, Any]):
        self._validate_structure(config_dict)
        self._load_and_validate(config_dict)

    def _validate_structure(self, config_dict: Dict[str, Any]) -> None:
        missing = [s for s in _REQUIRED_SECTIONS if s not in config_dict]
        if missing:
            raise ConfigValidationError(
                "Missing required sections: "
                + ", ".join(missing)
                + f"\nExpected sections: {', '.join(sorted(_KNOWN_SECTIONS))}"
            )
        unknown = [s for s in config_dict.keys() if s not in _KNOWN_SECTIONS]
        if unknown:
            print(f"WARNING: Unknown config sections (ignored): {', '.join(sorted(unknown))}")

    def _load_and_validate(self, config_dict: Dict[str, Any]) -> None:
        errors: List[str] = []

        # --- Experiment ---
        exp = config_dict.get("experiment", {})
        self.disease_type = exp.get("disease_type")
        if not self.disease_type:
            errors.append("experiment.disease_type is required")
        elif self.disease_type not in _ALLOWED_DISEASES:
            errors.append(f"experiment.disease_type must be one of {sorted(_ALLOWED_DISEASES)}")

        # --- Training ---
        train = config_dict.get("training", {})
        self.indi_epochs = int(train.get("indi_epochs", 0))
        if self.indi_epochs <= 0:
            errors.append("training.indi_epochs must be a positive integer")

        self.shared_epochs = int(train.get("shared_epochs", 0))
        if self.shared_epochs <= 0:
            errors.append("training.shared_epochs must be a positive integer")

        self.indi_batch = int(train.get("indi_batch", 1))
        if self.indi_batch <= 0:
            errors.append("training.indi_batch must be > 0")

        self.shared_batch = int(train.get("shared_batch", 1))
        if self.shared_batch <= 0:
            errors.append("training.shared_batch must be > 0")

        self.optimizer = str(train.get("optimizer", "adam")).lower()
        if self.optimizer not in _ALLOWED_OPTIMIZERS:
            errors.append(f"training.optimizer must be one of {sorted(_ALLOWED_OPTIMIZERS)}")

        # --- Pretraining (optional) ---
        pre = config_dict.get("pretraining", {})
        self.pretrain_enabled = bool(pre.get("enabled", False))
        self.pretrain_epochs = int(pre.get("epochs", 50))
        if self.pretrain_epochs <= 0:
            errors.append("pretraining.epochs must be a positive integer")

        # --- Cross-validation ---
        cv = config_dict.get("cross_validation", {})
        strategy = str(cv.get("strategy", "loocv")).lower()
        if strategy not in {"loocv", "kfold"}:
            errors.append("cross_validation.strategy must be 'loocv' or 'kfold'")
        if strategy == "loocv":
            self.cv_folds = None
        else:
            self.cv_folds = int(cv.get("folds", 10))
            if self.cv_folds < 2:
                errors.append("cross_validation.folds must be an integer >= 2")

        self.random_seed = int(cv.get("random_seed", 42))

        # --- Performance ---
        perf = config_dict.get("performance", {})
        self.parallel_modalities = bool(perf.get("parallel_modalities", False))
        self.use_gpu = bool(perf.get("use_gpu", False))

        # --- Early stopping (optional) ---
        es = config_dict.get("early_stopping", {})
        self.early_stopping_enabled = bool(es.get("enabled", False))
        self.early_stopping_patience = int(es.get("patience", 10))
        if self.early_stopping_patience <= 0:
            errors.append("early_stopping.patience must be a positive integer")
        try:
            self.early_stopping_min_delta = float(es.get("min_delta", 0.0001))
        except (TypeError, ValueError):
            errors.append("early_stopping.min_delta must be a non-negative number")
            self.early_stopping_min_delta = 0.0001
        if self.early_stopping_min_delta < 0:
            errors.append("early_stopping.min_delta must be >= 0")
        self.early_stopping_restore_best_weights = bool(es.get("restore_best_weights", True))

        # --- Preprocessing (optional) ---
        prep = config_dict.get("preprocessing", {})
        self.pca_enabled = bool(prep.get("pca_enabled", False))
        try:
            self.pca_variance_threshold = float(prep.get("pca_variance_threshold", 0.95))
        except (TypeError, ValueError):
            errors.append("preprocessing.pca_variance_threshold must be a number in (0, 1]")
            self.pca_variance_threshold = 0.95
        if not (0.0 < self.pca_variance_threshold <= 1.0):
            errors.append("preprocessing.pca_variance_threshold must be in (0.0, 1.0]")
        self.pca_min_features = int(prep.get("pca_min_features", 1000))
        if self.pca_min_features <= 0:
            errors.append("preprocessing.pca_min_features must be a positive integer")

        # --- Data (optional) ---
        data = config_dict.get("data", {})
        self.use_metadata = bool(data.get("use_metadata", False))
        self.class_weighting = bool(data.get("class_weighting", False))

        # --- Paths (optional) ---
        paths = config_dict.get("paths", {})
        self.data_dir = str(paths.get("data_dir", "./data/{disease_type}"))
        self.logs_base_dir = str(paths.get("logs_base_dir", "./logs/{disease_type}/{timestamp}"))
        self.tmp_base_dir = str(paths.get("tmp_base_dir", "./tmp/{disease_type}"))
        self.output_base_dir = str(paths.get("output_base_dir", "./output/{disease_type}/{timestamp}"))

        if errors:
            raise ConfigValidationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    def __repr__(self) -> str:
        return (
            "Config(\n"
            f"  disease_type={self.disease_type},\n"
            f"  indi_epochs={self.indi_epochs}, shared_epochs={self.shared_epochs}, "
            f"pretrain_epochs={self.pretrain_epochs}, pretrain_enabled={self.pretrain_enabled},\n"
            f"  indi_batch={self.indi_batch}, shared_batch={self.shared_batch}, optimizer={self.optimizer},\n"
            f"  cv_folds={self.cv_folds}, random_seed={self.random_seed},\n"
            f"  parallel_modalities={self.parallel_modalities}, use_gpu={self.use_gpu},\n"
            f"  early_stopping_enabled={self.early_stopping_enabled}, patience={self.early_stopping_patience}, "
            f"min_delta={self.early_stopping_min_delta}, restore_best_weights={self.early_stopping_restore_best_weights},\n"
            f"  pca_enabled={self.pca_enabled}, pca_variance_threshold={self.pca_variance_threshold}, "
            f"pca_min_features={self.pca_min_features},\n"
            f"  use_metadata={self.use_metadata}, class_weighting={self.class_weighting}\n"
            ")"
        )


# ============================== Load from TOML ==========================

def load_config(config_path: str) -> Config:
    """
    Load and validate a TOML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Generate a template with: python ./src/main.py --init-config"
        )
    
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
    return Config(config_dict)


# ======================= Immutable Structured View ======================

@dataclass(frozen=True)
class ExperimentConfig:
    disease_type: str

@dataclass(frozen=True)
class TrainingConfig:
    indi_epochs: int
    shared_epochs: int
    indi_batch: int
    shared_batch: int
    optimizer: str  # "adam" | "sgd"

@dataclass(frozen=True)
class PretrainingConfig:
    enabled: bool
    epochs: int

@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    min_delta: float
    restore_best_weights: bool

@dataclass(frozen=True)
class CrossValidationConfig:
    strategy: str   # "loocv" | "kfold"
    folds: int      # if loocv, kept at 10 for display/compat
    random_seed: int

@dataclass(frozen=True)
class PerformanceConfig:
    parallel_modalities: bool
    use_gpu: bool

@dataclass(frozen=True)
class PreprocessingConfig:
    pca_enabled: bool
    pca_variance_threshold: float
    pca_min_features: int

@dataclass(frozen=True)
class DataConfig:
    use_metadata: bool
    class_weighting: bool

@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    logs_base_dir: str
    tmp_base_dir: str
    output_base_dir: str

@dataclass(frozen=True)
class StructuredConfig:
    """Immutable, nested view used by the rest of the app."""
    experiment: ExperimentConfig
    training: TrainingConfig
    pretraining: PretrainingConfig
    early_stopping: EarlyStoppingConfig
    cross_validation: CrossValidationConfig
    performance: PerformanceConfig
    preprocessing: PreprocessingConfig
    data: DataConfig
    paths: PathsConfig


def build_structured_config(cfg: Config) -> StructuredConfig:
    """
    Adapt validated Config into an immutable StructuredConfig.
    """
    strategy = "loocv" if cfg.cv_folds is None else "kfold"
    folds = 10 if cfg.cv_folds is None else cfg.cv_folds

    return StructuredConfig(
        experiment=ExperimentConfig(disease_type=cfg.disease_type),
        training=TrainingConfig(
            indi_epochs=cfg.indi_epochs,
            shared_epochs=cfg.shared_epochs,
            indi_batch=cfg.indi_batch,
            shared_batch=cfg.shared_batch,
            optimizer=cfg.optimizer,
        ),
        pretraining=PretrainingConfig(
            enabled=cfg.pretrain_enabled,
            epochs=cfg.pretrain_epochs,
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=cfg.early_stopping_enabled,
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            restore_best_weights=cfg.early_stopping_restore_best_weights,
        ),
        cross_validation=CrossValidationConfig(
            strategy=strategy,
            folds=folds,
            random_seed=cfg.random_seed,
        ),
        performance=PerformanceConfig(
            parallel_modalities=cfg.parallel_modalities,
            use_gpu=cfg.use_gpu,
        ),
        preprocessing=PreprocessingConfig(
            pca_enabled=cfg.pca_enabled,
            pca_variance_threshold=cfg.pca_variance_threshold,
            pca_min_features=cfg.pca_min_features,
        ),
        data=DataConfig(
            use_metadata=cfg.use_metadata,
            class_weighting=cfg.class_weighting,
        ),
        paths=PathsConfig(
            data_dir=cfg.data_dir,
            logs_base_dir=cfg.logs_base_dir,
            tmp_base_dir=cfg.tmp_base_dir,
            output_base_dir=cfg.output_base_dir,
        ),
    )


def load_structured_config(config_path: str) -> StructuredConfig:
    """
    Convenience: load TOML -> Config (validated) -> StructuredConfig (immutable).
    """
    return build_structured_config(load_config(config_path))
