#!/usr/bin/env python3
import os
import subprocess
import sys

CONFIG_NAMES = [
    "meta-only",
    "pretrain-only",
    "meta-pretrain",
]

DISEASES = [
    "T2D",
    "IBD",
    "CRC",
    "cirrhosis",
]

BASE_CMD = ["python", "m4dl-1"]
CONFIG_DIR = "configs"
RUNS = 10


def main():
    # Verify all config files exist first
    missing = []
    for disease in DISEASES:
        for name in CONFIG_NAMES:
            path = os.path.join(CONFIG_DIR, disease, f"{name}.toml")
            if not os.path.exists(path):
                missing.append(path)

    if missing:
        print("Missing config files:")
        for m in missing:
            print("  -", m)
        sys.exit(1)

    print("All config files verified.\n")

    # Run each repetition through all combinations
    for i in range(1, RUNS + 1):
        print(f"=== Repetition {i}/{RUNS} ===")
        for disease in DISEASES:
            for name in CONFIG_NAMES:
                config_path = os.path.join(CONFIG_DIR, disease, f"{name}.toml")
                print(f"  Running {disease}-{name}")
                cmd = BASE_CMD + ["-c", config_path]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    {disease}-{name} failed (exit code {e.returncode})")
                    # Uncomment the next line if you want to stop on first failure
                    # break
        print()


if __name__ == "__main__":
    main()
