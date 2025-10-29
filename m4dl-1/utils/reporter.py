import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from . import logger


def print_summary(individual_results, shared_results=None):
    """Print summary of all training results."""
    logger.log("\n" + "="*60)
    logger.log("TRAINING SUMMARY")
    logger.log("="*60)

    # Individual results
    logger.log("\nIndividual Models:")
    logger.log("-" * 60)
    logger.log(f"{'Model':<20} {'Acc':<8} {'TP':<6} {'FN':<6} {'FP':<6} {'TN':<6}")
    logger.log("-" * 60)

    for res in individual_results:
        logger.log(f"{res['name']:<20} {res['accuracy']:<8.2f} "
                   f"{res['TP']:<6} {res['FN']:<6} {res['FP']:<6} {res['TN']:<6}")

    # Average individual performance
    avg_acc = sum(r['accuracy'] for r in individual_results) / len(individual_results)
    avg_tp = sum(r['TP'] for r in individual_results) / len(individual_results)
    avg_fn = sum(r['FN'] for r in individual_results) / len(individual_results)
    avg_fp = sum(r['FP'] for r in individual_results) / len(individual_results)
    avg_tn = sum(r['TN'] for r in individual_results) / len(individual_results)

    logger.log("-" * 60)
    logger.log(f"{'Average':<20} {avg_acc:<8.2f} "
               f"{avg_tp:<6.1f} {avg_fn:<6.1f} {avg_fp:<6.1f} {avg_tn:<6.1f}")

    # Shared results
    if shared_results:
        logger.log("\nShared Model:")
        logger.log("-" * 60)
        logger.log(f"{'Model':<20} {'Acc':<8} {'TP':<6} {'FN':<6} {'FP':<6} {'TN':<6}")
        logger.log("-" * 60)
        logger.log(f"{shared_results['name']:<20} {shared_results['accuracy']:<8.2f} "
                   f"{shared_results['TP']:<6} {shared_results['FN']:<6} "
                   f"{shared_results['FP']:<6} {shared_results['TN']:<6}")

        # Comparison
        logger.log("\nComparison:")
        logger.log("-" * 60)
        improvement = shared_results['accuracy'] - avg_acc
        logger.log(f"Shared vs Average Individual: {improvement:+.2f} accuracy")

    logger.log("="*60 + "\n")


def save_results_to_files(individual_results, shared_results, output_dir, disease_type, timestamp):
    """Save results to CSV and text files in configured output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save individual results as CSV
    df_individual = pd.DataFrame(individual_results)
    csv_individual = output_dir / "individual_results.csv"
    df_individual.to_csv(csv_individual, index=False)
    logger.log(f"Individual results saved to: {csv_individual}")

    # 2. Save shared results as CSV (if exists)
    if shared_results:
        df_shared = pd.DataFrame([shared_results])
        csv_shared = output_dir / "shared_results.csv"
        df_shared.to_csv(csv_shared, index=False)
        logger.log(f"Shared results saved to: {csv_shared}")

    # 3. Save combined summary as CSV
    all_results = individual_results.copy()
    if shared_results:
        all_results.append(shared_results)
    df_all = pd.DataFrame(all_results)
    csv_all = output_dir / "all_results.csv"
    df_all.to_csv(csv_all, index=False)
    logger.log(f"Combined results saved to: {csv_all}")

    # 4. Save as JSON for easy parsing
    json_data = {
        "disease_type": disease_type,
        "timestamp": timestamp,
        "individual_results": individual_results,
        "shared_results": shared_results,
    }
    json_file = output_dir / "results.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.log(f"JSON results saved to: {json_file}")

    # 5. Save human-readable summary as text file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Disease Type: {disease_type}\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        # Individual results
        f.write("Individual Models:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<20} {'Acc':<8} {'TP':<6} {'FN':<6} {'FP':<6} {'TN':<6}\n")
        f.write("-" * 60 + "\n")

        for res in individual_results:
            f.write(f"{res['name']:<20} {res['accuracy']:<8.2f} "
                   f"{res['TP']:<6} {res['FN']:<6} {res['FP']:<6} {res['TN']:<6}\n")

        # Average
        avg_acc = sum(r['accuracy'] for r in individual_results) / len(individual_results)
        avg_tp = sum(r['TP'] for r in individual_results) / len(individual_results)
        avg_fn = sum(r['FN'] for r in individual_results) / len(individual_results)
        avg_fp = sum(r['FP'] for r in individual_results) / len(individual_results)
        avg_tn = sum(r['TN'] for r in individual_results) / len(individual_results)

        f.write("-" * 60 + "\n")
        f.write(f"{'Average':<20} {avg_acc:<8.2f} "
               f"{avg_tp:<6.1f} {avg_fn:<6.1f} {avg_fp:<6.1f} {avg_tn:<6.1f}\n")

        # Shared results
        if shared_results:
            f.write("\nShared Model:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Model':<20} {'Acc':<8} {'TP':<6} {'FN':<6} {'FP':<6} {'TN':<6}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{shared_results['name']:<20} {shared_results['accuracy']:<8.2f} "
                   f"{shared_results['TP']:<6} {shared_results['FN']:<6} "
                   f"{shared_results['FP']:<6} {shared_results['TN']:<6}\n")

            # Comparison
            f.write("\nComparison:\n")
            f.write("-" * 60 + "\n")
            improvement = shared_results['accuracy'] - avg_acc
            f.write(f"Shared vs Average Individual: {improvement:+.2f} accuracy\n")

        f.write("="*60 + "\n")

    logger.log(f"Summary saved to: {summary_file}")
    logger.log(f"All results saved to: {output_dir}\n")
