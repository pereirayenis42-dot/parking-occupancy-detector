
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageOps

from alignment import align_images


ROOT = Path(__file__).resolve().parent.parent


def load_photo(path):
    image = ImageOps.exif_transpose(
        Image.open(path)
    ).convert("RGB")

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


def crop_image(image, ratios):
    height, width = image.shape[:2]

    x1 = int(width * ratios[0])
    y1 = int(height * ratios[1])
    x2 = int(width * ratios[2])
    y2 = int(height * ratios[3])

    return image[y1:y2, x1:x2].copy()


def main():
    with open(ROOT / "data" / "parking_spaces.json") as f:
        config = json.load(f)

    reference_path = (
        ROOT / "data" / "raw" /
        config["reference_image"]
    )

    test_images = sorted(
        p for p in (ROOT / "data" / "test").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not test_images:
        print("No test photographs found.")
        return

    # Start with just one test photograph
    test_path = test_images[0]

    print(f"Reference: {reference_path.name}")
    print(f"Test: {test_path.name}")

    reference = load_photo(reference_path)
    test = load_photo(test_path)

    # Register full photographs before cropping
    aligned, diagnostics, _ = align_images(
        reference, test
    )

    print("\nALIGNMENT DIAGNOSTICS")

    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    # Crop the aligned image using reference coordinates
    crop = crop_image(
        aligned, config["crop_ratios"]
    )

    # Draw mapped parking spaces
    for space_id, corners in config["spaces"].items():
        polygon = np.array(corners, dtype=np.int32)

        cv2.polylines(
            crop, [polygon], True,
            (255, 255, 0), 2
        )

        center = np.mean(
            polygon, axis=0
        ).astype(int)

        cv2.putText(
            crop,
            space_id,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )

    output_folder = ROOT / "results" / "images"
    output_folder.mkdir(
        parents=True, exist_ok=True
    )

    output_path = (
        output_folder /
        f"{test_path.stem}_aligned_preview.jpg"
    )

    cv2.imwrite(str(output_path), crop)

    print(f"\nSaved aligned preview: {output_path}")


if __name__ == "__main__":
    main()
