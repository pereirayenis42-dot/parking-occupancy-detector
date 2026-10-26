
from pathlib import Path
import json

import cv2
import numpy as np

from alignment import align_images
from test_alignment import load_photo, crop_image


ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = "parking_2026-10-07_1807"

LANDMARK_FILE = (
    ROOT / "results" / "metrics" / "detect_first"
    / f"{IMAGE_NAME}_multi_landmark.json"
)

OUTPUT_DIR = (
    ROOT / "results" / "images"
    / "translation_preview"
)


def estimate_translation(data):
    offsets = []

    for measurements in data["spaces"].values():
        reference = np.asarray(
            measurements["reference_points"],
            dtype=np.float64
        )

        aligned = np.asarray(
            measurements["aligned_points"],
            dtype=np.float64
        )

        offsets.extend(reference[:2] - aligned[:2])

    return np.mean(offsets, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(LANDMARK_FILE) as f:
        landmark_data = json.load(f)

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
        / landmark_data["image"]
    )

    aligned, diagnostics, homography = (
        align_images(reference, test)
    )

    dx, dy = estimate_translation(landmark_data)

    # Apply translation to the already aligned image.
    translation_matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    height, width = aligned.shape[:2]

    corrected = cv2.warpAffine(
        aligned,
        translation_matrix,
        (width, height)
    )

    ratios = config["crop_ratios"]

    reference_crop = crop_image(reference, ratios)
    original_crop = crop_image(aligned, ratios)
    corrected_crop = crop_image(corrected, ratios)

    before_blend = cv2.addWeighted(
        reference_crop, 0.5,
        original_crop, 0.5, 0
    )

    after_blend = cv2.addWeighted(
        reference_crop, 0.5,
        corrected_crop, 0.5, 0
    )

    comparison = np.hstack([
        reference_crop,
        before_blend,
        after_blend
    ])

    output_file = (
        OUTPUT_DIR
        / f"{IMAGE_NAME}_comparison.jpg"
    )

    cv2.imwrite(str(output_file), comparison)

    print("\nTRANSLATION PREVIEW COMPLETE")
    print("=" * 50)
    print(f"Translation: dx={dx:.2f}, dy={dy:.2f}")
    print("\nPanels:")
    print("LEFT: Reference")
    print("MIDDLE: Before correction")
    print("RIGHT: After correction")
    print("\nSaved:", output_file)
    print("\nNo existing files were overwritten.")


if __name__ == "__main__":
    main()
