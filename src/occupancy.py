
import cv2
import numpy as np
from abc import ABC, abstractmethod


class Confidence(ABC):
    """Base interface for parking-space confidence scoring."""

    @abstractmethod
    def score(self, overlap_ratio: float, box_confidence: float) -> float:
        """Return a scalar confidence value for a detection."""

    @abstractmethod
    def is_confident(self, score: float, threshold: float) -> bool:
        """Return whether a score should count as a valid detection."""


class DetectionConfidence(Confidence):
    """Combine detector confidence with spatial overlap into a useful score."""

    def score(self, overlap_ratio: float, box_confidence: float) -> float:
        overlap_ratio = max(0.0, min(1.0, float(overlap_ratio)))
        box_confidence = max(0.0, min(1.0, float(box_confidence)))

        # Penalize weak overlap, but still honor strong detector confidence.
        return box_confidence * overlap_ratio

    def is_confident(self, score: float, threshold: float) -> bool:
        return float(score) >= float(threshold)


def classify_occupancy(
    yolo_result,
    parking_spaces,
    overlap_threshold=0.15
):
    """
    A space is occupied when a detected vehicle:
    1. Has sufficient overlap with the parking polygon.
    2. Has its bounding-box center inside the polygon.
    """

    status = {
        space_id: "VACANT"
        for space_id in parking_spaces
    }

    boxes = yolo_result.boxes

    if boxes is None or len(boxes) == 0:
        return status

    height, width = yolo_result.orig_shape

    for space_id, corners in parking_spaces.items():
        polygon = np.array(corners, dtype=np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)

        area = np.count_nonzero(mask)

        if area == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Vehicle bounding-box center
            center = (
                float((x1 + x2) / 2),
                float((y1 + y2) / 2)
            )

            inside = cv2.pointPolygonTest(
                polygon,
                center,
                False
            ) >= 0

            if not inside:
                continue

            # Clamp bounding-box coordinates
            bx1 = max(0, int(x1))
            by1 = max(0, int(y1))
            bx2 = min(width, int(np.ceil(x2)))
            by2 = min(height, int(np.ceil(y2)))

            if bx2 <= bx1 or by2 <= by1:
                continue

            overlap_area = np.count_nonzero(
                mask[by1:by2, bx1:bx2]
            )

            overlap = overlap_area / area

            if overlap >= overlap_threshold:
                status[space_id] = "OCCUPIED"

                print(
                    f"{space_id}: OCCUPIED "
                    f"| overlap={overlap:.3f} "
                    f"| center={center}"
                )

                break

        if status[space_id] == "VACANT":
            print(f"{space_id}: VACANT")

    return status

def draw_occupancy(image, parking_spaces, status):
    """
    Draw occupancy polygons and labels.
    Input and output are PIL RGB images.
    """

    from PIL import Image

    output = cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

    overlay = output.copy()

    for space_id, corners in parking_spaces.items():

        polygon = np.array(
            corners, dtype=np.int32
        )

        occupied = status[space_id] == "OCCUPIED"

        # OpenCV uses BGR colors
        color = (
            (0, 0, 255) if occupied
            else (0, 180, 0)
        )

        cv2.fillPoly(
            overlay, [polygon], color
        )

        cv2.polylines(
            output, [polygon], True, color, 2
        )

        center = np.mean(
            polygon, axis=0
        ).astype(int)

        cv2.putText(
            output,
            space_id,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )

    # Blend transparent color overlays
    output = cv2.addWeighted(
        overlay, 0.25, output, 0.75, 0
    )

    return Image.fromarray(
        cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    )
