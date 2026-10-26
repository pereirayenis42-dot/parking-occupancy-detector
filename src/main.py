
from pathlib import Path
import json

from PIL import Image, ImageOps

from detector import VehicleDetector
from occupancy import classify_occupancy, draw_occupancy


def main():
    project_root = Path(__file__).resolve().parent.parent

    config_path = project_root / "data" / "parking_spaces.json"
    data_folder = project_root / "data" / "raw"
    output_folder = project_root / "results" / "images"

    output_folder.mkdir(parents=True, exist_ok=True)

    # Load saved parking-space configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    parking_spaces = config["spaces"]
    crop_ratios = config["crop_ratios"]

    # Use the same reference photo used to map spaces
    image_path = data_folder / config["reference_image"]

    image = ImageOps.exif_transpose(
        Image.open(image_path)
    ).convert("RGB")

    width, height = image.size

    # Recreate the exact crop from the mapper
    crop_bounds = (
        int(width * crop_ratios[0]),
        int(height * crop_ratios[1]),
        int(width * crop_ratios[2]),
        int(height * crop_ratios[3])
    )

    parking_crop = image.crop(crop_bounds)

    # Verify coordinate-system compatibility
    expected_size = tuple(config["crop_size"])

    if parking_crop.size != expected_size:
        raise ValueError(
            f"Crop mismatch: {parking_crop.size} "
            f"does not match mapped size {expected_size}"
        )

    print("=" * 45)
    print("PARKING OCCUPANCY DETECTOR")
    print("=" * 45)
    print(f"Image: {image_path.name}")
    print(f"Crop size: {parking_crop.size}")

    # Detect vehicles in the cropped image
    detector = VehicleDetector(
        model_path=str(project_root / "yolo11n.pt")
    )

    result = detector.detect(
        parking_crop,
        confidence=0.25
    )

    import cv2
    import numpy as np

    # Start with YOLO's annotated bounding boxes
    debug_image = result.plot()

    # Draw parking-space outlines
    for space_id, corners in parking_spaces.items():
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
            0.45,
            (255, 255, 255),
            2
        )

    # Draw the matching point for every vehicle
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            point = (
                int((x1 + x2) / 2),
                int(y2)
            )

            cv2.circle(
                debug_image,
                point,
                5,
                (0, 0, 255),
                -1
            )

    debug_path = output_folder / "debug_matching.jpg"

    cv2.imwrite(str(debug_path), debug_image)

    print(f"Debug image saved: {debug_path}")
    print(f"Vehicles detected: {len(result.boxes)}")

    # Automatically predict occupancy
    predicted_status = classify_occupancy(
        result, parking_spaces
    )

    # Draw parking-space predictions
    output = draw_occupancy(
        parking_crop,
        parking_spaces,
        predicted_status
    )

    # Save output
    output_path = (
        output_folder / "occupancy_result.jpg"
    )

    output.save(output_path)

    print("\nPARKING OCCUPANCY RESULTS")
    print("-" * 35)

    for space_id, status in predicted_status.items():
        print(f"{space_id}: {status}")

    print(f"\nSaved result: {output_path}")


if __name__ == "__main__":
    main()
