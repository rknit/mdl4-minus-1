# MDL4(-1)

Multi-modal deep learning for microbiome disease classification based on MDL4Microbiome. Check out the original MDL4 [repository](https://github.com/DMnBI/MDL4Microbiome)!

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python m4dl-1 -c configs/dev.toml
```

## Usage

```bash
# Create config template
python m4dl-1 --init-config configs/my-config.toml

# Run with config
python m4dl-1 -c configs/my-config.toml
```

## Configuration

Three presets available in `configs/`:
- **baseline.toml** - Exact MDL4 replication
- **dev.toml** - Fast iteration (k-fold, parallel)
- **prod.toml** - Maximum accuracy (LOOCV, all features)

## Data Format

Place data in `data/{disease}/`:
- `datasets.txt` - Modality file list
- `metadata.csv` - Covariates (optional)
- `ylab.txt` - One-hot labels
- `modality_*.csv.gz` - Sparse feature data

## Output

Results saved to `output/{disease}/{timestamp}/`:
- `individual_results.csv` - Per-modality accuracies
- `shared_results.csv` - Shared model accuracy
- `all_results.csv` - Combined results
- `results.json` - Structured JSON
- `summary.txt` - Human-readable summary

