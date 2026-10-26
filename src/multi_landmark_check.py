
from pathlib import Path
import json

import cv2
import numpy as np
import sys

from alignment import align_images
from test_alignment import load_photo, crop_image


ROOT = Path(__file__).resolve().parent.parent
TEST_FILENAME = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "parking_2026-10-07_1807.jpg"
)

SPACE_IDS = ["A01", "A02", "A03", "A04"]
MIN_LANDMARKS = 3


def select_points(image, title, count):
    """Click exactly count points in a displayed image."""

    points = []
    display = image.copy()

    def redraw():
        preview = display.copy()

        for i, (x, y) in enumerate(points, start=1):
            cv2.circle(
                preview, (x, y), 4, (0, 255, 255), -1
            )

            cv2.putText(
                preview, str(i), (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

        cv2.imshow(title, preview)

    def on_click(event, x, y, flags, userdata):
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(points) < count
        ):
            points.append((x, y))
            redraw()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    redraw()

    print(f"\n{title}")
    print(f"Click {count} fixed landmarks.")
    print("Press BACKSPACE to undo a click.")
    print("Press ENTER after selecting all points.")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (8, 127) and points:
            points.pop()
            redraw()

        elif key in (13, 10):
            if len(points) == count:
                break

            print(
                f"Please select {count} points. "
                f"Currently selected: {len(points)}"
            )

    cv2.destroyWindow(title)

    return points


def calculate_statistics(reference_pts, aligned_pts):
    reference_pts = np.asarray(
        reference_pts, dtype=np.float32
    )

    aligned_pts = np.asarray(
        aligned_pts, dtype=np.float32
    )

    errors = np.linalg.norm(
        reference_pts - aligned_pts,
        axis=1
    )

    median = float(np.median(errors))

    mad = float(
        np.median(np.abs(errors - median))
    )

    return {
        "errors_px": errors.tolist(),
        "median_error_px": median,
        "maximum_error_px": float(np.max(errors)),
        "mad_px": mad,
        "landmark_count": len(errors)
    }


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

    results = {}

    for space_id in SPACE_IDS:
        print("\n" + "=" * 50)
        print(f"SELECT LANDMARKS FOR {space_id}")
        print("=" * 50)

        reference_points = select_points(
            reference_crop,
            f"{space_id} - Reference",
            MIN_LANDMARKS
        )

        aligned_points = select_points(
            aligned_crop,
            f"{space_id} - Aligned",
            MIN_LANDMARKS
        )

        statistics = calculate_statistics(
            reference_points,
            aligned_points
        )

        results[space_id] = {
            "reference_points": reference_points,
            "aligned_points": aligned_points,
            **statistics
        }

        print(
            f"{space_id}: "
            f"median={statistics['median_error_px']:.2f} px "
            f"| max={statistics['maximum_error_px']:.2f} px "
            f"| MAD={statistics['mad_px']:.2f} px"
        )

    output_folder = (
        ROOT / "results" / "metrics" / "detect_first"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_folder /
        f"{Path(TEST_FILENAME).stem}_multi_landmark.json"
    )

    with open(output_path, "w") as f:
        json.dump(
            {
                "image": TEST_FILENAME,
                "minimum_landmarks": MIN_LANDMARKS,
                "spaces": results
            },
            f,
            indent=2
        )

    print("\n" + "=" * 50)
    print("MULTI-LANDMARK ANALYSIS COMPLETE")
    print("=" * 50)

    for space_id, data in results.items():
        print(
            f"{space_id}: "
            f"{data['landmark_count']} landmarks "
            f"| median={data['median_error_px']:.2f} px "
            f"| max={data['maximum_error_px']:.2f} px "
            f"| MAD={data['mad_px']:.2f} px"
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
