
from pathlib import Path
import json

import cv2
import numpy as np

from alignment import align_images
from test_alignment import load_photo, crop_image


ROOT = Path(__file__).resolve().parent.parent

TEST_FILENAME = "parking_2026-10-07_1807.jpg"

reference_points = []
aligned_points = []


def collect_points(window_name, image, points):
    display = image.copy()

    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            cv2.circle(
                display,
                (x, y),
                5,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                display,
                str(len(points)),
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)
    cv2.imshow(window_name, display)

    print(f"\nClick landmarks in: {window_name}")
    print("Press ENTER when finished.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            break

    cv2.destroyWindow(window_name)


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

    print("\nMANUAL LANDMARK ALIGNMENT CHECK")
    print("=" * 45)
    print("Select the SAME landmarks in the SAME order.")
    print("Use fixed pavement features only.")

    collect_points(
        "Reference - Click landmarks",
        reference_crop,
        reference_points
    )

    collect_points(
        "Aligned - Click same landmarks",
        aligned_crop,
        aligned_points
    )

    if len(reference_points) != len(aligned_points):
        raise ValueError(
            "Reference and aligned point counts differ"
        )

    if not reference_points:
        raise ValueError("No landmarks selected")

    ref = np.array(
        reference_points, dtype=np.float32
    )

    aligned_pts = np.array(
        aligned_points, dtype=np.float32
    )

    errors = np.linalg.norm(
        ref - aligned_pts,
        axis=1
    )

    print("\nLANDMARK DISPLACEMENTS")
    print("=" * 45)

    for index, error in enumerate(errors, 1):
        print(
            f"Landmark {index}: {error:.2f} px"
        )

    print(
        f"\nMedian displacement: "
        f"{np.median(errors):.2f} px"
    )

    print(
        f"Maximum displacement: "
        f"{np.max(errors):.2f} px"
    )

    output = (
        ROOT / "results" / "metrics" /
        "manual_landmark_quality.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output, "w") as f:
        json.dump(
            {
                "image": TEST_FILENAME,
                "reference_points": reference_points,
                "aligned_points": aligned_points,
                "errors_px": errors.tolist(),
                "median_error_px": float(
                    np.median(errors)
                ),
                "maximum_error_px": float(
                    np.max(errors)
                )
            },
            f,
            indent=2
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
