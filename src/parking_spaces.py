
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "results" / "images"
CONFIG_PATH = PROJECT_ROOT / "data" / "parking_spaces.json"

# Keep the same crop proportions used in Colab
CROP_RATIOS = (0.20, 0.43, 0.65, 0.72)

spaces = {}
current_points = []
preview = None
base_image = None


def draw_preview():
    global preview

    preview = base_image.copy()

    # Draw completed spaces
    for space_id, corners in spaces.items():
        polygon = np.array(corners, dtype=np.int32)

        cv2.polylines(
            preview, [polygon], True,
            (255, 255, 0), 2
        )

        center = np.mean(polygon, axis=0).astype(int)

        cv2.putText(
            preview, space_id, tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 2
        )

    # Draw current clicks
    for point in current_points:
        cv2.circle(
            preview, tuple(point),
            4, (0, 0, 255), -1
        )

    if len(current_points) > 1:
        cv2.polylines(
            preview,
            [np.array(current_points, dtype=np.int32)],
            False,
            (0, 0, 255),
            2
        )

    cv2.imshow("Parking Space Mapper", preview)


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    current_points.append([x, y])
    print(f"Corner {len(current_points)}: ({x}, {y})")

    if len(current_points) == 4:
        space_id = f"A{len(spaces) + 1:02d}"
        spaces[space_id] = current_points.copy()
        current_points.clear()
        print(f"Saved {space_id}")

    draw_preview()


def main():
    global base_image

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in DATA_FOLDER.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not images:
        print("No images found in data/raw/")
        return

    # Use first photograph as reference
    image_path = images[0]
    print(f"Reference image: {image_path.name}")

    image = ImageOps.exif_transpose(
        Image.open(image_path)
    ).convert("RGB")

    w, h = image.size
    ratios = CROP_RATIOS

    bounds = (
        int(w * ratios[0]),
        int(h * ratios[1]),
        int(w * ratios[2]),
        int(h * ratios[3])
    )

    cropped = image.crop(bounds)
    base_image = cv2.cvtColor(
        np.array(cropped), cv2.COLOR_RGB2BGR
    )

    cv2.namedWindow("Parking Space Mapper")
    cv2.setMouseCallback("Parking Space Mapper", on_mouse)

    print("\nClick 4 corners for each parking space.")
    print("Press S to save, U to undo, Q to quit.")

    draw_preview()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            if current_points:
                print("Finish or undo incomplete points first.")
                continue

            CONFIG_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )

            config = {
                "reference_image": image_path.name,
                "reference_size": [w, h],
                "crop_ratios": list(CROP_RATIOS),
                "crop_size": [
                    cropped.width, cropped.height
                ],
                "spaces": spaces
            }

            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=4)

            print(f"Saved: {CONFIG_PATH}")

            cv2.imwrite(
                str(OUTPUT_FOLDER / "parking_map.jpg"),
                preview
            )

        elif key == ord("u"):
            if current_points:
                current_points.pop()
            elif spaces:
                last_id = list(spaces)[-1]
                del spaces[last_id]
            draw_preview()

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
