
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

    all_reference = []
    all_aligned = []

    for space_id, measurements in data["spaces"].items():
        all_reference.extend(
            measurements["reference_points"]
        )

        all_aligned.extend(
            measurements["aligned_points"]
        )

    reference = np.asarray(
        all_reference, dtype=np.float64
    )

    aligned = np.asarray(
        all_aligned, dtype=np.float64
    )

    # A single global translation that minimizes
    # squared landmark displacement.
    translation = np.mean(
        reference - aligned,
        axis=0
    )

    corrected = aligned + translation

    before = np.linalg.norm(
        reference - aligned,
        axis=1
    )

    after = np.linalg.norm(
        reference - corrected,
        axis=1
    )

    print("\nGLOBAL TRANSLATION EXPERIMENT")
    print("=" * 50)

    print(
        f"Translation: dx={translation[0]:.2f}px, "
        f"dy={translation[1]:.2f}px"
    )

    print(
        f"Median before: {np.median(before):.2f}px"
    )

    print(
        f"Median after:  {np.median(after):.2f}px"
    )

    print("\nPER-SPACE COMPARISON")
    print("=" * 50)

    index = 0

    for space_id, measurements in data["spaces"].items():
        count = measurements["landmark_count"]

        space_before = before[index:index + count]
        space_after = after[index:index + count]

        print(
            f"{space_id}: "
            f"before={np.median(space_before):.2f}px "
            f"| after={np.median(space_after):.2f}px"
        )

        index += count

    print(
        "\nDiagnostic only: no registration "
        "or occupancy files were changed."
    )


if __name__ == "__main__":
    main()
