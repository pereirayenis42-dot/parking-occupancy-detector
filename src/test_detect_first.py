
from pathlib import Path
import json

import cv2
import numpy as np

from alignment import align_images
from test_alignment import load_photo, crop_image


ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "data" / "parking_spaces.json") as f:
        config = json.load(f)

    reference_path = (
        ROOT / "data" / "raw" /
        config["reference_image"]
    )

    test_path = (
        ROOT / "data" / "test" /
        "parking_2026-10-06_1442.jpg"
    )

    reference = load_photo(reference_path)
    test = load_photo(test_path)

    # Register geometry, but preserve original image
    aligned, diagnostics, homography = align_images(
        reference, test
    )

    # Detect vehicles on original photograph
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "yolo11n.pt"))

    result = model.predict(
        source=test,
        conf=0.10,
        imgsz=1280,
        classes=[2, 3, 5, 7],
        device="cpu",
        verbose=False
    )[0]

    # Draw transformed detections on aligned crop
    crop = crop_image(
        aligned, config["crop_ratios"]
    )

    ref_height, ref_width = reference.shape[:2]

    crop_x = int(
        ref_width * config["crop_ratios"][0]
    )
    crop_y = int(
        ref_height * config["crop_ratios"][1]
    )

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Four corners of original YOLO box
        points = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]).reshape(-1, 1, 2)

        # Transform original coordinates to reference
        transformed = cv2.perspectiveTransform(
            points, homography
        ).reshape(-1, 2)

        # Convert to crop-local coordinates
        transformed[:, 0] -= crop_x
        transformed[:, 1] -= crop_y

        polygon = np.round(
            transformed
        ).astype(np.int32)

        cv2.polylines(
            crop,
            [polygon],
            True,
            (0, 255, 255),
            2
        )

    # Draw reference parking polygons
    for space_id, corners in config["spaces"].items():
        polygon = np.array(corners, dtype=np.int32)

        cv2.polylines(
            crop,
            [polygon],
            True,
            (255, 255, 0),
            2
        )

        center = np.mean(
            polygon, axis=0
        ).astype(int)

        cv2.putText(
            crop,
            space_id,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    output_path = (
        ROOT / "results" / "images" /
        "detect_first_preview.jpg"
    )

    cv2.imwrite(str(output_path), crop)

    print("Original detections:", len(result.boxes))
    print("Alignment diagnostics:", diagnostics)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
