
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = "parking_2026-10-07_1807"

QUALITY_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_quality_assessment.json"
)

OUTPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_quality_gate.json"
)


def apply_quality_gate(prediction, quality_status):
    """
    Return occupancy only when registration
    has been independently verified.

    REVIEW, INSUFFICIENT, and UNVERIFIED
    are treated as UNKNOWN.
    """

    if quality_status != "VERIFIED_ACCEPTABLE":
        return "UNKNOWN"

    if prediction not in {"OCCUPIED", "VACANT"}:
        return "UNKNOWN"

    return prediction


def main():
    with open(QUALITY_FILE) as f:
        quality_report = json.load(f)

    results = {}

    for space_id, quality in quality_report["spaces"].items():
        results[space_id] = {
            "quality_status": quality["status"],
            "gated_occupancy": apply_quality_gate(
                prediction=None,
                quality_status=quality["status"]
            ),
            "reason": quality["reason"]
        }

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "image": quality_report["image"],
                "mode": "experimental_quality_gate",
                "note": (
                    "No occupancy predictions were loaded. "
                    "This demonstrates quality-based abstention."
                ),
                "spaces": results
            },
            f,
            indent=2
        )

    print("\nQUALITY-GATED OCCUPANCY EXPERIMENT")
    print("=" * 55)

    for space_id, result in results.items():
        print(
            f"{space_id}: "
            f"quality={result['quality_status']} "
            f"| result={result['gated_occupancy']}"
        )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
