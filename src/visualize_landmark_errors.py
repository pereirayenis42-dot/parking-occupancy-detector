
from pathlib import Path
import json

import cv2
import numpy as np

from alignment import align_images
from test_alignment import load_photo, crop_image


ROOT = Path(__file__).resolve().parent.parent
IMAGE_NAME = "parking_2026-10-07_1807"

INPUT_JSON = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_multi_landmark.json"
)

OUTPUT_DIR = (
    ROOT / "results" / "images"
    / "registration_quality"
)


def draw_landmark_arrows(image, measurements, space_id):
    canvas = image.copy()

    reference_points = measurements["reference_points"]
    aligned_points = measurements["aligned_points"]

    for index, (ref_pt, aligned_pt) in enumerate(
        zip(reference_points, aligned_points), start=1
    ):
        ref_xy = tuple(map(int, ref_pt))
        aligned_xy = tuple(map(int, aligned_pt))

        error = float(
            np.linalg.norm(
                np.asarray(ref_pt, dtype=float)
                - np.asarray(aligned_pt, dtype=float)
            )
        )

        # Reference landmark: green
        cv2.circle(
            canvas, ref_xy, 5, (0, 255, 0), -1
        )

        # Aligned landmark: red
        cv2.circle(
            canvas, aligned_xy, 5, (0, 0, 255), -1
        )

        # Arrow points from aligned location
        # toward the reference location.
        cv2.arrowedLine(
            canvas,
            aligned_xy,
            ref_xy,
            (0, 255, 255),
            2,
            tipLength=0.2
        )

        label = f"{space_id}-{index}: {error:.1f}px"

        label_x = min(
            ref_xy[0] + 8,
            canvas.shape[1] - 120
        )
        label_y = max(
            20,
            ref_xy[1] - 8
        )

        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1
        )

    return canvas


def main():
    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    with open(INPUT_JSON) as f:
        data = json.load(f)

    with open(
        ROOT / "data" / "parking_spaces.json"
    ) as f:
        config = json.load(f)

    reference = load_photo(
        ROOT / "data" / "raw"
        / config["reference_image"]
    )

    test = load_photo(
        ROOT / "data" / "test"
        / data["image"]
    )

    aligned, diagnostics, homography = (
        align_images(reference, test)
    )

    reference_crop = crop_image(
        reference, config["crop_ratios"]
    )

    aligned_crop = crop_image(
        aligned, config["crop_ratios"]
    )

    # Blend images to reveal registration differences.
    base = cv2.addWeighted(
        reference_crop, 0.5,
        aligned_crop, 0.5, 0
    )

    combined = base.copy()

    for space_id, measurements in data["spaces"].items():

        single = draw_landmark_arrows(
            base, measurements, space_id
        )

        output_file = (
            OUTPUT_DIR
            / f"{IMAGE_NAME}_{space_id}_landmarks.jpg"
        )

        cv2.imwrite(str(output_file), single)

        combined = draw_landmark_arrows(
            combined, measurements, space_id
        )

    combined_file = (
        OUTPUT_DIR
        / f"{IMAGE_NAME}_all_landmarks.jpg"
    )

    cv2.imwrite(
        str(combined_file), combined
    )

    print("\nLANDMARK VISUALIZATION COMPLETE")
    print("=" * 50)

    for space_id in data["spaces"]:
        print(f"{space_id}: overlay saved")

    print("\nCombined image:")
    print(combined_file)


if __name__ == "__main__":
    main()
