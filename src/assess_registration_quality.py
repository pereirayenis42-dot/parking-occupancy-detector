
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = "parking_2026-10-07_1807"

INPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_multi_landmark.json"
)

OUTPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_quality_assessment.json"
)

MIN_LANDMARKS = 3

# Experimental investigation trigger, NOT a
# validated tolerance for parking occupancy.
REVIEW_TRIGGER_PX = 10.0


def assess_space(measurements):
    count = measurements["landmark_count"]
    median = measurements["median_error_px"]
    maximum = measurements["maximum_error_px"]
    mad = measurements["mad_px"]

    if count < MIN_LANDMARKS:
        status = "INSUFFICIENT"
        reason = "Fewer than three landmarks"

    elif median >= REVIEW_TRIGGER_PX:
        status = "REVIEW"
        reason = (
            "Median landmark displacement exceeds "
            "the experimental investigation trigger"
        )

    else:
        status = "UNVERIFIED"
        reason = (
            "Below investigation trigger, but "
            "registration accuracy is not validated"
        )

    return {
        "status": status,
        "reason": reason,
        "landmark_count": count,
        "median_error_px": median,
        "maximum_error_px": maximum,
        "mad_px": mad
    }


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    assessments = {
        space_id: assess_space(measurements)
        for space_id, measurements
        in data["spaces"].items()
    }

    report = {
        "image": data["image"],
        "review_trigger_px": REVIEW_TRIGGER_PX,
        "trigger_is_validated": False,
        "method": "manual stationary landmark comparison",
        "spaces": assessments
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("\nREGISTRATION QUALITY ASSESSMENT")
    print("=" * 55)

    for space_id, result in assessments.items():
        print(
            f"{space_id}: {result['status']} "
            f"| median={result['median_error_px']:.2f} px "
            f"| MAD={result['mad_px']:.2f} px"
        )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
