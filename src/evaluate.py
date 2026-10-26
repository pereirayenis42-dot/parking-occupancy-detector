
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    ROOT / "results" / "metrics" /
    "validation_predictions.csv"
)

GROUND_TRUTH_PATH = (
    ROOT / "data" / "ground_truth.csv"
)

OUTPUT_FOLDER = ROOT / "results" / "metrics"

LABELS = {"OCCUPIED", "VACANT"}


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def main():
    predictions = read_csv(PREDICTIONS_PATH)
    ground_truth = read_csv(GROUND_TRUTH_PATH)

    # Match records using image filename and space ID
    truth_lookup = {}

    for row in ground_truth:
        key = (row["image"].strip(), row["space_id"].strip())

        if key in truth_lookup:
            raise ValueError(
                f"Duplicate ground-truth label: {key}"
            )

        truth_lookup[key] = row

    tp = tn = fp = fn = 0
    evaluated = []
    skipped = []
    seen_predictions = set()

    for row in predictions:
        image = row["image"].strip()
        space_id = row["space_id"].strip()
        predicted = row["predicted"].strip().upper()

        key = (image, space_id)

        if key in seen_predictions:
            raise ValueError(
                f"Duplicate prediction: {key}"
            )

        seen_predictions.add(key)

        truth = truth_lookup.get(key)

        if truth is None:
            skipped.append({
                "image": image,
                "space_id": space_id,
                "reason": "Missing ground truth"
            })
            continue

        actual = truth["actual"].strip().upper()
        alignment = truth["alignment_ok"].strip().upper()

        if alignment != "YES":
            skipped.append({
                "image": image,
                "space_id": space_id,
                "reason": "Alignment not approved"
            })
            continue

        if actual not in LABELS or predicted not in LABELS:
            skipped.append({
                "image": image,
                "space_id": space_id,
                "reason": "Missing or invalid label"
            })
            continue

        if actual == "OCCUPIED" and predicted == "OCCUPIED":
            outcome = "TP"
            tp += 1

        elif actual == "VACANT" and predicted == "VACANT":
            outcome = "TN"
            tn += 1

        elif actual == "VACANT" and predicted == "OCCUPIED":
            outcome = "FP"
            fp += 1

        else:
            outcome = "FN"
            fn += 1

        evaluated.append({
            "image": image,
            "space_id": space_id,
            "actual": actual,
            "predicted": predicted,
            "outcome": outcome
        })

    total = tp + tn + fp + fn

    if total == 0:
        print("No verified predictions available to evaluate.")
        return

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    metrics = {
        "evaluated_spaces": total,
        "skipped_spaces": len(skipped),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FOLDER / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(
        OUTPUT_FOLDER / "evaluation_details.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "space_id", "actual",
                "predicted", "outcome"
            ]
        )
        writer.writeheader()
        writer.writerows(evaluated)

    with open(
        OUTPUT_FOLDER / "evaluation_skipped.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "space_id", "reason"]
        )
        writer.writeheader()
        writer.writerows(skipped)

    print("\n" + "=" * 45)
    print("PARKING OCCUPANCY EVALUATION")
    print("=" * 45)

    print(f"Evaluated spaces: {total}")
    print(f"Skipped spaces:   {len(skipped)}")
    print(f"True positives:   {tp}")
    print(f"True negatives:   {tn}")
    print(f"False positives:  {fp}")
    print(f"False negatives:  {fn}")

    print("\nPERFORMANCE METRICS")
    print("-" * 35)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 score:  {f1:.2%}")

    print("\nEvaluation files saved to results/metrics/")


if __name__ == "__main__":
    main()
