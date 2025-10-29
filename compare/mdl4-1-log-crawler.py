#!/usr/bin/env python

from pathlib import Path
from dataclasses import dataclass
import traceback

ROOT_PATH = Path("./mdl4-1-runs/logs")


# csv header:
#   disease, kind, timestamp,
#   modal1_acc, modal1_TP, modal1_FN, modal1_FP, modal1_TN,
#   modal2_acc, modal2_TP, modal2_FN, modal2_FP, modal2_TN,
#   shared_acc, shared_TP, shared_FN, shared_FP, shared_TN
@dataclass(frozen=True)
class LogEntry:
    disease: str
    kind: str
    timestamp: str
    modal1_acc: float
    modal1_TP: int
    modal1_FN: int
    modal1_FP: int
    modal1_TN: int
    modal2_acc: float
    modal2_TP: int
    modal2_FN: int
    modal2_FP: int
    modal2_TN: int
    shared_acc: float
    shared_TP: int
    shared_FN: int
    shared_FP: int
    shared_TN: int


def read_entry(log_file_path: Path) -> LogEntry:
    # extract csv from path: ./mdl4-1-runs/logs/{disease}/{kind}/{timestamp}/main.log
    parts = log_file_path.parts
    disease = parts[-4]
    kind = parts[-3]
    timestamp = parts[-2]

    modal1_acc = modal1_TP = modal1_FN = modal1_FP = modal1_TN = None
    modal2_acc = modal2_TP = modal2_FN = modal2_FP = modal2_TN = None

    with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        for line in lines:
            # order: Model Acc TP FN FP TN (space separated)
            if "modality_1_species_sparse" in line:
                tokens = line.strip().split()
                modal1_acc = float(tokens[1])
                modal1_TP = int(tokens[2])
                modal1_FN = int(tokens[3])
                modal1_FP = int(tokens[4])
                modal1_TN = int(tokens[5])
            if "modality_2_pathways_sparse" in line:
                tokens = line.strip().split()
                modal2_acc = float(tokens[1])
                modal2_TP = int(tokens[2])
                modal2_FN = int(tokens[3])
                modal2_FP = int(tokens[4])
                modal2_TN = int(tokens[5])
                break

        shared_stat_start = 0
        for i, line in enumerate(lines):
            if "Shared Model:" in line:
                shared_stat_start = i + 4
                break

        tokens = lines[shared_stat_start].strip().split()
        shared_acc = float(tokens[1])
        shared_TP = int(tokens[2])
        shared_FN = int(tokens[3])
        shared_FP = int(tokens[4])
        shared_TN = int(tokens[5])

    assert modal1_acc is not None, f"modal1_acc not found in {log_file_path}"
    assert modal2_acc is not None, f"modal2_acc not found in {log_file_path}"
    assert shared_acc is not None, f"shared_acc not found in {log_file_path}"

    return LogEntry(
        disease=disease,
        kind=kind,
        timestamp=timestamp,
        modal1_acc=modal1_acc,
        modal1_TP=modal1_TP,
        modal1_TN=modal1_TN,
        modal1_FP=modal1_FP,
        modal1_FN=modal1_FN,
        modal2_acc=modal2_acc,
        modal2_TP=modal2_TP,
        modal2_TN=modal2_TN,
        modal2_FP=modal2_FP,
        modal2_FN=modal2_FN,
        shared_acc=shared_acc,
        shared_TP=shared_TP,
        shared_TN=shared_TN,
        shared_FP=shared_FP,
        shared_FN=shared_FN,
    )


def main():
    entries = []
    error_logs = []

    for disease in ROOT_PATH.iterdir():
        if not disease.is_dir():
            print(f"Skipping non-directory: {disease}")
            continue
        for kind in disease.iterdir():
            if not kind.is_dir():
                print(f"Skipping non-directory: {kind}")
                continue
            for timestamp in kind.iterdir():
                if not timestamp.is_dir():
                    print(f"Skipping non-directory: {timestamp}")
                    continue
                log_file = timestamp / "main.log"
                if not log_file.exists():
                    print(f"Log file not found: {log_file}")
                    continue
                try:
                    entry = read_entry(log_file)
                    entries.append(entry)
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")
                    error_logs.append((log_file, str(e)))
                    # Uncomment below if you want to see full traceback
                    # traceback.print_exc()
                    continue  # skip this log and move to next

    # Write results
    with open("mdl4-1-runs.csv", "w") as f:
        f.write(
            "disease,kind,timestamp,"
            "modal1_acc,modal1_TP,modal1_FN,modal1_FP,modal1_TN,"
            "modal2_acc,modal2_TP,modal2_FN,modal2_FP,modal2_TN,"
            "shared_acc,shared_TP,shared_FN,shared_FP,shared_TN\n"
        )
        for entry in entries:
            f.write(
                f"{entry.disease},{entry.kind},{entry.timestamp},"
                f"{entry.modal1_acc},{entry.modal1_TP},{entry.modal1_FN},{entry.modal1_FP},{entry.modal1_TN},"
                f"{entry.modal2_acc},{entry.modal2_TP},{entry.modal2_FN},{entry.modal2_FP},{entry.modal2_TN},"
                f"{entry.shared_acc},{entry.shared_TP},{entry.shared_FN},{entry.shared_FP},{entry.shared_TN}\n"
            )

    # Write error report
    if error_logs:
        with open("mdl4-1-errors.txt", "w") as ef:
            for path, err in error_logs:
                ef.write(f"{path}: {err}\n")
        print(
            f"\n{len(error_logs)} logs had errors. See mdl4-1-errors.txt for details."
        )


if __name__ == "__main__":
    main()
