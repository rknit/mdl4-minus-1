import pandas as pd
import numpy as np
from os import path
from typing import Optional
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse


def get_dataset_path(
    disease_type: str, dataset_name: str, data_dir: Optional[str] = None
) -> str:
    """
    Get path to a dataset file.

    Args:
        disease_type: Disease type (e.g., "T2D", "CRC")
        dataset_name: Name of the dataset file
        data_dir: Base data directory (defaults to "./data/{disease_type}")

    Returns:
        Full path to the dataset file

    Raises:
        FileNotFoundError: If the file does not exist
    """
    disease_type = disease_type.strip() if disease_type else ""
    dataset_name = dataset_name.strip() if dataset_name else ""

    assert disease_type, "disease_type cannot be empty"
    assert dataset_name, "dataset_name cannot be empty"

    if data_dir is None:
        data_dir = f"./data/{disease_type}"

    fixed_path = f"{data_dir}/{dataset_name}"
    if not path.isfile(fixed_path):
        raise FileNotFoundError(f"{dataset_name} does not exist in {data_dir}")
    return fixed_path


def get_dataset_paths(disease_type: str, data_dir: Optional[str] = None) -> list[str]:
    """
    Get paths to all modality dataset files.

    Args:
        disease_type: Disease type (e.g., "T2D", "CRC")
        data_dir: Base data directory (defaults to "./data/{disease_type}")

    Returns:
        List of full paths to modality dataset files
    """
    modality_path = get_dataset_path(disease_type, "datasets.txt", data_dir)

    dp = []
    with open(modality_path, "r") as f:
        dataset_paths = f.read().splitlines()
        for line_num, dataset_path in enumerate(dataset_paths, start=1):
            dataset_path = dataset_path.strip()
            if not dataset_path or dataset_path.startswith("#"):
                continue
            try:
                fixed_path = get_dataset_path(disease_type, dataset_path, data_dir)
                dp.append(fixed_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Line {line_num} in datasets.txt: {e}") from e

    assert dp, f"No valid dataset paths found in {modality_path}"
    return dp


def get_metadata(disease_type: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load metadata CSV file.

    Args:
        disease_type: Disease type (e.g., "T2D", "CRC")
        data_dir: Base data directory (defaults to "./data/{disease_type}")

    Returns:
        DataFrame containing metadata
    """
    metadata_path = get_dataset_path(disease_type, "metadata.csv", data_dir)
    df = pd.read_csv(metadata_path, sep=",")

    assert not df.empty, f"Metadata file {metadata_path} is empty"

    return df


CONTROL_ALIASES = {"control", "ctrl", "healthy", "normal"}


def load_labels(disease_type: str, data_dir: Optional[str] = None) -> np.ndarray:
    """
    Returns one-hot labels with fixed mapping:
      control -> 0 -> [1, 0]
      non-control (i.e., disease) -> 1 -> [0, 1]

    Args:
        disease_type: Disease type (e.g., "T2D", "CRC")
        data_dir: Base data directory (defaults to "./data/{disease_type}")

    Returns:
        One-hot encoded label array
    """
    label_file = get_dataset_path(disease_type, "ylab.txt", data_dir)
    with open(label_file, "r") as f:
        raw = [line.strip() for line in f if line.strip()]

    assert raw, f"Label file {label_file} contains no valid labels"

    # Normalize -> ints (0=control, 1=disease)
    ints = []
    for s in raw:
        t = s.lower()
        ints.append(0 if t in CONTROL_ALIASES else 1)

    y_int = np.asarray(ints, dtype=np.int64).reshape(-1, 1)

    # One-hot with stable category order [0,1]
    enc = OneHotEncoder(
        categories=[[0, 1]],  # type: ignore[arg-type]
        sparse_output=False,
        dtype=np.float32,  # type: ignore[arg-type]
    )

    y_onehot = enc.fit_transform(y_int).astype(np.float32)
    return y_onehot


def load_data_for_disease(disease_type: str, data_dir: Optional[str] = None):
    """
    Load all data for a disease type (modality paths, metadata, labels).

    Args:
        disease_type: Disease type (e.g., "T2D", "CRC")
        data_dir: Base data directory (defaults to "./data/{disease_type}")

    Returns:
        Tuple of (modality_paths, metadata_df, labels_array)
    """
    modality_paths = get_dataset_paths(disease_type, data_dir)
    metadata_df = get_metadata(disease_type, data_dir)
    labels_array = load_labels(disease_type, data_dir)

    n_labels = len(labels_array)
    n_metadata = len(metadata_df)

    assert n_labels == n_metadata, (
        f"Label count ({n_labels}) does not match metadata count ({n_metadata})"
    )

    return modality_paths, metadata_df, labels_array


def load_sparse_triplet_csv_gz(datapath: str) -> np.ndarray:
    df = pd.read_csv(
        datapath,
        compression="gzip",
        dtype={"row": int, "col": int, "value": np.float32},
    )

    required_cols = ["row", "col", "value"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(
            f"{datapath} missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"{datapath} contains no data (only header)")

    if (df["row"] < 1).any() or (df["col"] < 1).any():
        raise ValueError(
            f"{datapath} contains invalid indices (must be >= 1). "
            f"Row range: [{df['row'].min()}, {df['row'].max()}], "
            f"Col range: [{df['col'].min()}, {df['col'].max()}]"
        )

    if df["value"].isna().any():  # type: ignore[attr-defined]
        raise ValueError(f"{datapath} contains NaN values")

    if np.isinf(df["value"].to_numpy()).any():
        raise ValueError(f"{datapath} contains infinite values")

    n_rows = int(df["row"].max())
    n_cols = int(df["col"].max())

    # reconstruct sparse feature matrix (features x samples)
    mat_sparse = sparse.coo_matrix(
        (df["value"], (df["row"].to_numpy() - 1, df["col"].to_numpy() - 1)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )

    # transpose so samples are rows
    mat_dense = mat_sparse.T.toarray().astype(np.float32)

    assert mat_dense.shape[0] == n_cols, "Transposition error: sample count mismatch"
    assert mat_dense.shape[1] == n_rows, "Transposition error: feature count mismatch"

    return mat_dense
