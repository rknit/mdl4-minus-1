"""Training execution for modalities (sequential and parallel)."""

from typing import List
from models import Modality, ModalityResults


def train_modalities_sequential(modalities: List[Modality]) -> List[ModalityResults]:
    """Train modalities sequentially."""
    results: List[ModalityResults] = []
    for modality in modalities:
        modality.initialize()
        result = modality.train()
        results.append(result)
        modality.cleanup()
    return results


def train_modalities_parallel(modalities: List[Modality]) -> List[ModalityResults]:
    """Train modalities in parallel using joblib."""
    from joblib import Parallel, delayed

    def train_one(modality: Modality) -> ModalityResults:
        modality.initialize()
        result = modality.train()
        modality.cleanup()
        return result

    n_jobs = len(modalities)
    results = list(
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(train_one)(mod) for mod in modalities
        )
    )
    return results  # type: ignore
