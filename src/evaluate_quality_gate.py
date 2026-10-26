
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent
IMAGE_NAME = "parking_2026-10-07_1807"

INPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_gated_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_gate_metrics.json"
)


def main():
    with open(INPUT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("No gated predictions found")

    total = len(rows)

    accepted = [
        row for row in rows
        if row["gated_prediction"] in {"OCCUPIED", "VACANT"}
    ]

    unknown = [
        row for row in rows
        if row["gated_prediction"] == "UNKNOWN"
    ]

    if len(accepted) + len(unknown) != total:
        raise ValueError("Unexpected gated prediction values")

    coverage = len(accepted) / total
    abstention_rate = len(unknown) / total

    metrics = {
        "image": IMAGE_NAME + ".jpg",
        "total_predictions": total,
        "accepted_predictions": len(accepted),
        "unknown_predictions": len(unknown),
        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_accuracy": None,
        "selective_accuracy_note": (
            "Not calculated: requires ground-truth labels "
            "for accepted predictions. Undefined when "
            "no predictions are accepted."
        )
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nQUALITY GATE EVALUATION")
    print("=" * 50)

    print(f"Total predictions: {total}")
    print(f"Accepted predictions: {len(accepted)}")
    print(f"UNKNOWN predictions: {len(unknown)}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Abstention rate: {abstention_rate:.2%}")
    print("Selective accuracy: N/A")

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
