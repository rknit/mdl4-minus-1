from .preprocessing import PCAStrategy, NoOpPreprocessingStrategy
from .pretraining import AutoencoderStrategy, NoOpPretrainingStrategy
from .metadata import ConcatenateMetadataStrategy, NoOpMetadataStrategy
from .early_stopping import CallbackEarlyStoppingStrategy, NoOpEarlyStoppingStrategy
from .class_weighting import BalancedWeightStrategy, NoOpWeightStrategy


class StrategyFactory:
    """Factory for creating strategy instances from config."""

    @staticmethod
    def create_preprocessing(config):
        """Create preprocessing strategy from config."""
        if config.preprocessing.pca_enabled:
            return PCAStrategy(
                variance_threshold=config.preprocessing.pca_variance_threshold,
                min_features=config.preprocessing.pca_min_features,
            )
        return NoOpPreprocessingStrategy()

    @staticmethod
    def create_pretraining(config):
        """Create pretraining strategy from config."""
        if config.pretraining.enabled:
            return AutoencoderStrategy(
                epochs=config.pretraining.epochs,
                batch_size=config.training.indi_batch,
                optimizer=config.training.optimizer,
            )
        return NoOpPretrainingStrategy()

    @staticmethod
    def create_metadata(config):
        """Create metadata strategy from config."""
        if config.data.use_metadata:
            return ConcatenateMetadataStrategy()
        return NoOpMetadataStrategy()

    @staticmethod
    def create_early_stopping(config):
        """Create early stopping strategy from config."""
        if config.early_stopping.enabled:
            return CallbackEarlyStoppingStrategy(
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                restore_best_weights=config.early_stopping.restore_best_weights,
            )
        return NoOpEarlyStoppingStrategy()

    @staticmethod
    def create_class_weighting(config):
        """Create class weighting strategy from config."""
        if config.data.class_weighting:
            return BalancedWeightStrategy()
        return NoOpWeightStrategy()
