
from pathlib import Path
import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageOps

from alignment import align_images
from detector import VehicleDetector
from occupancy import classify_occupancy, draw_occupancy


ROOT = Path(__file__).resolve().parent.parent


def load_photo(path):
    """Load a photograph with correct EXIF orientation."""
    image = ImageOps.exif_transpose(
        Image.open(path)
    ).convert("RGB")

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


def crop_image(image, ratios):
    """Apply the reference photograph's crop ratios."""
    height, width = image.shape[:2]

    x1 = int(width * ratios[0])
    y1 = int(height * ratios[1])
    x2 = int(width * ratios[2])
    y2 = int(height * ratios[3])

    return image[y1:y2, x1:x2].copy()


def draw_parking_map(image, parking_spaces):
    """Draw parking-space outlines for alignment review."""
    output = image.copy()

    for space_id, corners in parking_spaces.items():
        polygon = np.array(corners, dtype=np.int32)

        cv2.polylines(
            output, [polygon], True,
            (255, 255, 0), 2
        )

        center = np.mean(polygon, axis=0).astype(int)

        cv2.putText(
            output, space_id, tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 2
        )

    return output


def main():
    test_folder = ROOT / "data" / "test"
    output_folder = (
        ROOT / "results" / "images" / "validation"
    )
    metrics_folder = ROOT / "results" / "metrics"

    output_folder.mkdir(parents=True, exist_ok=True)
    metrics_folder.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "data" / "parking_spaces.json") as f:
        config = json.load(f)

    spaces = config["spaces"]
    ratios = config["crop_ratios"]

    reference_path = (
        ROOT / "data" / "raw" /
        config["reference_image"]
    )

    reference = load_photo(reference_path)

    detector = VehicleDetector(
        model_path=str(ROOT / "yolo11n.pt")
    )

    test_images = sorted(
        p for p in test_folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not test_images:
        print("No test photographs found.")
        return

    prediction_records = []
    alignment_records = []

    for image_path in test_images:
        print("\n" + "=" * 50)
        print(f"Processing: {image_path.name}")

        try:
            test_image = load_photo(image_path)

            # Diagnostic: detect vehicles BEFORE image registration
            if image_path.name == "parking_2026-10-06_1442.jpg":
                original_result = detector.model.predict(
                    source=test_image,
                    conf=0.10,
                    imgsz=1280,
                    classes=[2, 3, 5, 7],
                    device="cpu",
                    verbose=False
                )[0]

                original_debug = original_result.plot()

                cv2.imwrite(
                    str(
                        output_folder /
                        f"{image_path.stem}_original_debug.jpg"
                    ),
                    original_debug
                )

                print(
                    "Original photograph detections:",
                    len(original_result.boxes)
                )

            # 1. Register full photograph
            aligned, diagnostics, _ = align_images(
                reference, test_image
            )

            # 2. Crop registered photograph
            crop = crop_image(aligned, ratios)

            expected_size = tuple(config["crop_size"])

            if crop.shape[1::-1] != expected_size:
                raise ValueError(
                    "Aligned crop dimensions do not match "
                    "the reference crop."
                )

            # 3. Save alignment preview
            alignment_preview = draw_parking_map(
                crop, spaces
            )

            cv2.imwrite(
                str(
                    output_folder /
                    f"{image_path.stem}_alignment.jpg"
                ),
                alignment_preview
            )

            # 4. Convert crop for YOLO11
            crop_rgb = cv2.cvtColor(
                crop, cv2.COLOR_BGR2RGB
            )

            pil_crop = Image.fromarray(crop_rgb)

            # 5. Run vehicle detection
            result = detector.detect(
                pil_crop,
                confidence=0.25
            )

            # Save YOLO11 bounding boxes for debugging
            debug_image = result.plot()

            # Draw parking-space polygons over detections
            for space_id, corners in spaces.items():
                polygon = np.array(corners, dtype=np.int32)

                cv2.polylines(
                    debug_image,
                    [polygon],
                    True,
                    (255, 255, 0),
                    2
                )

                center = np.mean(polygon, axis=0).astype(int)

                cv2.putText(
                    debug_image,
                    space_id,
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            cv2.imwrite(
                str(
                    output_folder /
                    f"{image_path.stem}_debug.jpg"
                ),
                debug_image
            )

            # Diagnostic only: inspect low-confidence detections
            if image_path.name == "parking_2026-10-06_1442.jpg":
                low_conf_result = detector.detect(
                    pil_crop,
                    confidence=0.10
                )

                low_conf_debug = low_conf_result.plot()

                for space_id, corners in spaces.items():
                    polygon = np.array(corners, dtype=np.int32)

                    cv2.polylines(
                        low_conf_debug,
                        [polygon],
                        True,
                        (255, 255, 0),
                        2
                    )

                    center = np.mean(
                        polygon, axis=0
                    ).astype(int)

                    cv2.putText(
                        low_conf_debug,
                        space_id,
                        tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

                cv2.imwrite(
                    str(
                        output_folder /
                        f"{image_path.stem}_low_conf_debug.jpg"
                    ),
                    low_conf_debug
                )

            # 6. Predict occupancy
            predictions = classify_occupancy(
                result, spaces
            )

            # 7. Save annotated result
            annotated = draw_occupancy(
                pil_crop, spaces, predictions
            )

            annotated.save(
                output_folder /
                f"{image_path.stem}_occupancy.jpg"
            )

            # 8. Record predictions
            for space_id, prediction in predictions.items():
                prediction_records.append({
                    "image": image_path.name,
                    "space_id": space_id,
                    "predicted": prediction,
                    "actual": "",
                    "alignment_ok": ""
                })

            alignment_records.append({
                "image": image_path.name,
                "status": "PROCESSED",
                **diagnostics
            })

            print("Predictions:", predictions)
            print("Alignment:", diagnostics)

            print("Predictions:", predictions)
            print("Alignment:", diagnostics)

        except Exception as error:
            print(f"FAILED: {error}")

            alignment_records.append({
                "image": image_path.name,
                "status": "FAILED",
                "error": str(error)
            })

    # Save predictions
    predictions_path = (
        metrics_folder / "validation_predictions.csv"
    )

    with open(predictions_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "space_id",
                "predicted",
                "actual",
                "alignment_ok"
            ]
        )

        writer.writeheader()
        writer.writerows(prediction_records)

    # Save alignment diagnostics
    alignment_path = (
        metrics_folder / "alignment_diagnostics.csv"
    )

    alignment_fields = [
        "image",
        "status",
        "reference_features",
        "test_features",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "error"
    ]

    with open(alignment_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=alignment_fields
        )

        writer.writeheader()
        writer.writerows(alignment_records)

    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Photographs attempted: {len(test_images)}")
    print(f"Photographs processed: {len(alignment_records) - sum(r['status'] == 'FAILED' for r in alignment_records)}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Alignment diagnostics: {alignment_path}")


if __name__ == "__main__":
    main()
