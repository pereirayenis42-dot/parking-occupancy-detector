
from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = "parking_2026-10-07_1807"

PREDICTIONS_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / "detect_first_predictions.csv"
)

QUALITY_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_quality_assessment.json"
)

OUTPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_gated_predictions.csv"
)


def normalize_image_name(value):
    return Path(value).name


def gate_prediction(prediction, quality):
    prediction = str(prediction).strip().upper()

    if quality != "VERIFIED_ACCEPTABLE":
        return "UNKNOWN"

    if prediction not in {"OCCUPIED", "VACANT"}:
        return "UNKNOWN"

    return prediction


def main():
    with open(QUALITY_FILE) as f:
        quality_report = json.load(f)

    target_image = normalize_image_name(
        quality_report["image"]
    )

    with open(PREDICTIONS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("Prediction CSV has no header")

    print("\nAVAILABLE CSV COLUMNS:")
    print(fieldnames)

    # Expected schema from our validation pipeline.
    required = {"image", "space_id", "predicted"}

    missing = required - set(fieldnames)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            "Check the printed CSV headers."
        )

    output_rows = []

    for row in rows:
        if normalize_image_name(row["image"]) != target_image:
            continue

        space_id = row["space_id"].strip()

        if space_id not in quality_report["spaces"]:
            continue

        quality = quality_report["spaces"][space_id]
        quality_status = quality["status"]

        updated = dict(row)

        updated["registration_quality"] = quality_status

        updated["gated_prediction"] = gate_prediction(
            row["predicted"],
            quality_status
        )

        output_rows.append(updated)

    if not output_rows:
        raise ValueError(
            f"No matching predictions found for {target_image}"
        )

    output_fields = list(fieldnames)

    for name in (
        "registration_quality",
        "gated_prediction"
    ):
        if name not in output_fields:
            output_fields.append(name)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=output_fields
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nQUALITY-GATED YOLO PREDICTIONS")
    print("=" * 60)

    for row in output_rows:
        print(
            f"{row['space_id']}: "
            f"YOLO={row['predicted']} "
            f"| quality={row['registration_quality']} "
            f"| gated={row['gated_prediction']}"
        )

    print("\nSaved:", OUTPUT_FILE)
    print("Original prediction CSV was not modified.")


if __name__ == "__main__":
    main()
