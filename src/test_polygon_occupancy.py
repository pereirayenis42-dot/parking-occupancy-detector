
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from alignment import align_images
from test_alignment import load_photo, crop_image
from occupancy_polygons import (
    transform_vehicle_polygons,
    classify_polygon_occupancy
)


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

    aligned, diagnostics, homography = align_images(
        reference, test
    )

    model = YOLO(str(ROOT / "yolo11n.pt"))

    # Detect on original photograph
    result = model.predict(
        source=test,
        conf=0.10,
        imgsz=1280,
        classes=[2, 3, 5, 7],
        device="cpu",
        verbose=False
    )[0]

    # Reference crop offset
    ref_height, ref_width = reference.shape[:2]

    ratios = config["crop_ratios"]

    crop_x = int(ref_width * ratios[0])
    crop_y = int(ref_height * ratios[1])

    # Transform detections
    vehicles = transform_vehicle_polygons(
        result,
        homography,
        crop_x,
        crop_y
    )

    aligned_crop = crop_image(
        aligned,
        ratios
    )

    # Classify occupancy
    predictions = classify_polygon_occupancy(
        vehicles,
        config["spaces"],
        aligned_crop.shape
    )

    print("\n" + "=" * 40)
    print("DETECT-FIRST OCCUPANCY RESULTS")
    print("=" * 40)

    for space_id, status in predictions.items():
        print(f"{space_id}: {status}")

    # Save visualization
    output = aligned_crop.copy()

    # Yellow vehicle polygons
    for vehicle in vehicles:
        cv2.polylines(
            output,
            [vehicle["polygon"]],
            True,
            (0, 255, 255),
            2
        )

    # Parking-space polygons
    for space_id, corners in config["spaces"].items():
        polygon = np.array(
            corners, dtype=np.int32
        )

        color = (
            (0, 180, 0)
            if predictions[space_id] == "OCCUPIED"
            else (0, 0, 255)
        )

        cv2.polylines(
            output,
            [polygon],
            True,
            color,
            2
        )

        center = np.mean(
            polygon, axis=0
        ).astype(int)

        cv2.putText(
            output,
            f"{space_id}: {predictions[space_id]}",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    output_path = (
        ROOT / "results" / "images" /
        "polygon_occupancy_preview.jpg"
    )

    cv2.imwrite(str(output_path), output)

    print("\nAlignment:", diagnostics)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
