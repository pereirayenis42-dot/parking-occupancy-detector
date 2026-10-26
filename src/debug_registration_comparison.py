
from pathlib import Path
import json

import cv2
import numpy as np

from alignment import align_images
from test_alignment import load_photo, crop_image


ROOT = Path(__file__).resolve().parent.parent

# Change this to the photograph with the shadow
TEST_FILENAME = "parking_2026-10-07_1807.jpg"


def main():
    with open(ROOT / "data" / "parking_spaces.json") as f:
        config = json.load(f)

    reference = load_photo(
        ROOT / "data" / "raw" /
        config["reference_image"]
    )

    test = load_photo(
        ROOT / "data" / "test" /
        TEST_FILENAME
    )

    aligned, diagnostics, homography = align_images(
        reference, test
    )

    ratios = config["crop_ratios"]

    reference_crop = crop_image(reference, ratios)
    aligned_crop = crop_image(aligned, ratios)

    if reference_crop.shape != aligned_crop.shape:
        raise ValueError("Crop dimensions do not match")

    # Blend both images to expose displaced lines
    blend = cv2.addWeighted(
        reference_crop, 0.5,
        aligned_crop, 0.5,
        0
    )

    # Difference between reference and aligned image
    difference = cv2.absdiff(
        reference_crop,
        aligned_crop
    )

    # Create side-by-side comparison
    comparison = np.hstack([
        reference_crop,
        aligned_crop,
        blend,
        difference
    ])

    output = (
        ROOT / "results" / "images" /
        "registration_comparison.jpg"
    )

    cv2.imwrite(str(output), comparison)

    print("Alignment diagnostics:", diagnostics)
    print("Saved:", output)


if __name__ == "__main__":
    main()
