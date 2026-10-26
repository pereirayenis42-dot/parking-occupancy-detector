
import json
from pathlib import Path

import numpy as np


def evaluate_per_space_quality(
    landmark_json,
    space_order,
    review_threshold_px=20.0
):
    """
    Experimental per-space quality assessment.

    Assumes exactly one manually matched landmark
    for each parking space, in space_order.

    Threshold is a provisional review trigger,
    not a validated registration accuracy limit.
    """

    with open(landmark_json) as f:
        data = json.load(f)

    errors = data["errors_px"]

    if len(errors) != len(space_order):
        raise ValueError(
            "Landmark count must match space count"
        )

    results = {}

    for space_id, error in zip(space_order, errors):

        if error is None or not np.isfinite(error):
            status = "INSUFFICIENT"
            reason = "Missing or invalid measurement"

        elif error >= review_threshold_px:
            status = "REVIEW"
            reason = "Large landmark displacement"

        else:
            status = "PROVISIONAL"
            reason = "Below review threshold; not verified"

        results[space_id] = {
            "status": status,
            "error_px": (
                float(error)
                if error is not None
                and np.isfinite(error)
                else None
            ),
            "reason": reason,
            "landmark_count": 1
        }

    return results


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent.parent

    landmark_json = (
        ROOT / "results" / "metrics" /
        "manual_landmark_quality.json"
    )

    quality = evaluate_per_space_quality(
        landmark_json=landmark_json,
        space_order=["A01", "A02", "A03", "A04"]
    )

    print("\nPER-SPACE REGISTRATION QUALITY")
    print("=" * 50)

    for space_id, result in quality.items():
        print(
            f"{space_id}: {result['status']} "
            f"| error={result['error_px']:.2f} px "
            f"| {result['reason']}"
        )

    output_path = (
        ROOT / "results" / "metrics" /
        "per_space_quality.json"
    )

    with open(output_path, "w") as f:
        json.dump(quality, f, indent=2)

    print("\nSaved:", output_path)
