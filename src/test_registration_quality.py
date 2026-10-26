
from pathlib import Path
import json

from alignment import align_images
from test_alignment import load_photo
from registration_quality import check_registration_quality


ROOT = Path(__file__).resolve().parent.parent

# Use the shadowed photograph you identified
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

    quality = check_registration_quality(
        homography=homography,
        original_shape=test.shape,
        reference_shape=reference.shape,
        parking_spaces=config["spaces"],
        crop_ratios=config["crop_ratios"],
        minimum_coverage=0.90
    )

    print("\nREGISTRATION QUALITY")
    print("=" * 45)

    for space_id, result in quality.items():
        print(
            f"{space_id}: {result['status']} "
            f"| coverage={result['coverage']:.1%} "
            f"| {result['reason']}"
        )

    print("\nAlignment diagnostics:", diagnostics)


if __name__ == "__main__":
    main()
