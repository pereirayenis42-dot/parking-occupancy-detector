
from pathlib import Path
import csv
import json

import cv2
import numpy as np
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

    reference = load_photo(
        ROOT / "data" / "raw" /
        config["reference_image"]
    )

    test_folder = ROOT / "data" / "test"

    output_folder = (
        ROOT / "results" / "images" / "detect_first"
    )

    metrics_folder = (
        ROOT / "results" / "metrics" / "detect_first"
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    metrics_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ROOT / "yolo11n.pt"))

    ref_height, ref_width = reference.shape[:2]
    ratios = config["crop_ratios"]

    crop_x = int(ref_width * ratios[0])
    crop_y = int(ref_height * ratios[1])

    records = []

    images = sorted(
        p for p in test_folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for image_path in images:
        print(f"\nProcessing: {image_path.name}")

        try:
            test = load_photo(image_path)

            # Align geometry to reference
            aligned, diagnostics, homography = align_images(
                reference, test
            )

            # Detect on original image
            result = model.predict(
                source=test,
                conf=0.10,
                imgsz=1280,
                classes=[2, 3, 5, 7],
                device="cpu",
                verbose=False
            )[0]

            # Transform detections
            vehicles = transform_vehicle_polygons(
                result,
                homography,
                crop_x,
                crop_y
            )

            aligned_crop = crop_image(
                aligned, ratios
            )

            predictions = classify_polygon_occupancy(
                vehicles,
                config["spaces"],
                aligned_crop.shape
            )

            # Draw results
            output = aligned_crop.copy()

            for vehicle in vehicles:
                cv2.polylines(
                    output,
                    [vehicle["polygon"]],
                    True,
                    (0, 255, 255),
                    2
                )

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
                    output, [polygon], True, color, 2
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

                records.append({
                    "image": image_path.name,
                    "space_id": space_id,
                    "predicted": predictions[space_id]
                })

            cv2.imwrite(
                str(
                    output_folder /
                    f"{image_path.stem}_detect_first.jpg"
                ),
                output
            )

            print("Predictions:", predictions)
            print("Alignment:", diagnostics)

        except Exception as error:
            print(f"FAILED: {error}")

    output_csv = (
        metrics_folder / "detect_first_predictions.csv"
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "space_id", "predicted"]
        )
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 45)
    print("DETECT-FIRST VALIDATION COMPLETE")
    print("=" * 45)
    print(f"Photographs attempted: {len(images)}")
    print(f"Predictions recorded: {len(records)}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
