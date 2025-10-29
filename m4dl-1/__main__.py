#!/usr/bin/env python3

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import run_training
from utils.config import (
    ConfigValidationError,
    load_structured_config,
    write_template_to_file,
)


def main():
    parser = argparse.ArgumentParser(description="MDL4Microbiome(-1)")
    parser.add_argument("--init-config", type=str, help="Initialize config template")
    parser.add_argument("-c", "--config", type=str, help="Path to config TOML file")
    args = parser.parse_args()

    if args.init_config:
        write_template_to_file(args.init_config)
        print(f"Config template written to: {args.init_config}")
        return

    if not args.config:
        print("Error: --config argument required")
        print("Usage: python m4dl-1 -c <config.toml>")
        print("   or: python m4dl-1 --init-config <path.toml>")
        return

    try:
        config = load_structured_config(args.config)
    except ConfigValidationError as e:
        print(f"Config validation error: {e}")
        raise
    except FileNotFoundError as e:
        print(f"Config file not found: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

    run_training(config)


if __name__ == "__main__":
    main()
