
from pathlib import Path
import json

from alignment import align_images
from test_alignment import load_photo
from local_alignment_quality import evaluate_local_alignment


ROOT = Path(__file__).resolve().parent.parent

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

    quality = evaluate_local_alignment(
        reference=reference,
        test=test,
        homography=homography,
        parking_spaces=config["spaces"],
        crop_ratios=config["crop_ratios"]
    )

    print("\nLOCAL ALIGNMENT QUALITY")
    print("=" * 50)

    for space_id, result in quality.items():
        error = result["median_error"]

        error_text = (
            f"{error:.2f} px"
            if error is not None
            else "N/A"
        )

        print(
            f"{space_id}: {result['status']} "
            f"| matches={result['matches']} "
            f"| median error={error_text}"
        )

    print("\nAlignment diagnostics:", diagnostics)


if __name__ == "__main__":
    main()
