"""Main orchestration logic for training pipeline."""

from datetime import datetime
from pathlib import Path
from typing import List
import time

from context.training_context import TrainingContext
from context.modality_context import ModalityContext
from context.shared_context import SharedModelContext
from models import Modality, SharedModel, ModalityResults
from strategies.factory import StrategyFactory
from utils.loader import load_data_for_disease
from utils.config import StructuredConfig
from utils.path_resolver import create_resolver_from_config
from utils import logger
from .trainer import train_modalities_sequential, train_modalities_parallel


def create_training_context(config: StructuredConfig) -> TrainingContext:
    """Create and initialize global training context."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    resolver = create_resolver_from_config(
        disease_type=config.experiment.disease_type,
        timestamp=timestamp,
        cv_strategy=config.cross_validation.strategy,
    )

    resolved_paths = resolver.resolve_all(
        {
            "data_dir": config.paths.data_dir,
            "logs_base_dir": config.paths.logs_base_dir,
            "tmp_base_dir": config.paths.tmp_base_dir,
            "output_base_dir": config.paths.output_base_dir,
        }
    )

    datapaths, metadata, ylab = load_data_for_disease(
        config.experiment.disease_type, data_dir=resolved_paths["data_dir"]
    )

    print("Resolved directory paths:")
    print(f"  data_dir: {resolved_paths['data_dir']}")
    print(f"  logs_base_dir: {resolved_paths['logs_base_dir']}")
    print(f"  tmp_base_dir: {resolved_paths['tmp_base_dir']}")
    print(f"  output_base_dir: {resolved_paths['output_base_dir']}")

    context = TrainingContext(
        config=config,
        disease_type=config.experiment.disease_type,
        timestamp=timestamp,
        labels=ylab,
        metadata=metadata,
        cv_strategy=config.cross_validation.strategy,
        cv_folds=config.cross_validation.folds,
        random_seed=config.cross_validation.random_seed,
        use_gpu=config.performance.use_gpu,
        parallel_modalities=config.performance.parallel_modalities,
        data_dir=Path(resolved_paths["data_dir"]),
        logs_base_dir=Path(resolved_paths["logs_base_dir"]),
        tmp_base_dir=Path(resolved_paths["tmp_base_dir"]),
        output_base_dir=Path(resolved_paths["output_base_dir"]),
        datapaths=datapaths,
    )

    context.initialize()
    return context


def create_modalities(training_context: TrainingContext) -> List[Modality]:
    """Create Modality instances for each dataset."""
    modalities = []

    for datapath in training_context.datapaths:
        modality_name = Path(datapath).name.split(".")[0]

        context = ModalityContext(
            modality_name=modality_name,
            dataset_path=Path(datapath),
            epochs=training_context.config.training.indi_epochs,
            batch_size=training_context.config.training.indi_batch,
            optimizer=training_context.config.training.optimizer,
            bottleneck_dim=50,
            tmp_dir=training_context.tmp_base_dir / modality_name,
            log_file=training_context.logs_base_dir / f"{modality_name}.log",
            preprocessing_strategy=StrategyFactory.create_preprocessing(
                training_context.config
            ),
            pretraining_strategy=StrategyFactory.create_pretraining(
                training_context.config
            ),
            metadata_strategy=StrategyFactory.create_metadata(training_context.config),
            early_stopping_strategy=StrategyFactory.create_early_stopping(
                training_context.config
            ),
            class_weighting_strategy=StrategyFactory.create_class_weighting(
                training_context.config
            ),
        )

        modality = Modality(context, training_context)
        modalities.append(modality)

    return modalities


def create_shared_model_context(
    individual_results: List[ModalityResults], training_context: TrainingContext
) -> SharedModelContext:
    """Create shared model context from individual results."""
    return SharedModelContext(
        modality_activation_dirs=[Path(r.activation_path) for r in individual_results],
        features_per_modality=individual_results[0].features_per_individual,
        epochs=training_context.config.training.shared_epochs,
        batch_size=training_context.config.training.shared_batch,
        optimizer=training_context.config.training.optimizer,
        output_dir=training_context.output_base_dir,
        log_file=training_context.logs_base_dir / "shared.log",
        early_stopping_strategy=StrategyFactory.create_early_stopping(
            training_context.config
        ),
        class_weighting_strategy=StrategyFactory.create_class_weighting(
            training_context.config
        ),
    )


def run_training(config: StructuredConfig):
    """Main training pipeline."""
    print(f"Config: {config}")
    logger.log(f"Config: {config}")

    training_context = create_training_context(config)
    modalities = create_modalities(training_context)

    logger.log(f"Training {len(modalities)} modalities...")

    start_time = time.perf_counter()

    if config.performance.parallel_modalities:
        logger.log("Using parallel execution")
        individual_results = train_modalities_parallel(modalities)
    else:
        logger.log("Using sequential execution")
        individual_results = train_modalities_sequential(modalities)

    logger.log("Individual training completed, starting shared model training...")

    shared_context = create_shared_model_context(individual_results, training_context)
    shared_model = SharedModel(shared_context, training_context)
    shared_model.initialize()
    shared_results = shared_model.train(individual_results)

    logger.log("Shared model training completed, saving results...")
    shared_model.save_results(individual_results, shared_results)

    elapsed_time = time.perf_counter() - start_time

    msg = f"Total training time: {elapsed_time:.2f} seconds"
    print(msg, flush=True)
    logger.log(msg)

    shared_model.cleanup()
    training_context.cleanup()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Results saved to: {training_context.output_base_dir}")
    print(f"Logs saved to: {training_context.logs_base_dir}")
    print("=" * 80 + "\n")
