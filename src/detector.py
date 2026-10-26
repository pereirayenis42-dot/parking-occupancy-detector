
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

# COCO classes:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleDetector:
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.25):
        """Run YOLO11 on a PIL image."""

        results = self.model.predict(
            source=np.array(image.convert("RGB")),
            conf=confidence,
            classes=VEHICLE_CLASSES,
            device="cpu",
            verbose=False
        )

        return results[0]

    def save_preview(self, result, output_path):
        """Save an image with YOLO bounding boxes."""

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # YOLO plot() returns a BGR NumPy image
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        Image.fromarray(annotated_rgb).save(output_path)

        return output_path
