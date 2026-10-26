
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / "parking_2026-10-07_1807_multi_landmark.json"
)


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    training_offsets = []
    held_out = {}

    for space_id, measurements in data["spaces"].items():

        reference = np.asarray(
            measurements["reference_points"],
            dtype=np.float64
        )

        aligned = np.asarray(
            measurements["aligned_points"],
            dtype=np.float64
        )

        if len(reference) < 3:
            raise ValueError(
                f"{space_id} needs at least 3 landmarks"
            )

        # First two landmarks: estimate translation.
        training_offsets.extend(
            reference[:2] - aligned[:2]
        )

        # Third landmark: held-out validation.
        held_out[space_id] = {
            "reference": reference[2],
            "aligned": aligned[2]
        }

    translation = np.mean(
        np.asarray(training_offsets),
        axis=0
    )

    print("\nHELD-OUT TRANSLATION VALIDATION")
    print("=" * 55)

    print(
        f"Training translation: "
        f"dx={translation[0]:.2f}px, "
        f"dy={translation[1]:.2f}px"
    )

    before_errors = []
    after_errors = []

    for space_id, points in held_out.items():

        reference = points["reference"]
        aligned = points["aligned"]

        before = float(
            np.linalg.norm(reference - aligned)
        )

        after = float(
            np.linalg.norm(
                reference - (aligned + translation)
            )
        )

        before_errors.append(before)
        after_errors.append(after)

        change = "IMPROVED" if after < before else "WORSE"

        print(
            f"{space_id}: "
            f"before={before:.2f}px "
            f"| after={after:.2f}px "
            f"| {change}"
        )

    print("\nOVERALL HELD-OUT RESULTS")
    print("=" * 55)

    print(
        f"Median before: "
        f"{np.median(before_errors):.2f}px"
    )

    print(
        f"Median after:  "
        f"{np.median(after_errors):.2f}px"
    )

    print(
        "\nNo registration or occupancy files were modified."
    )


if __name__ == "__main__":
    main()
