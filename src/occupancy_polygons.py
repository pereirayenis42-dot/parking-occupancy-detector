
import cv2
import numpy as np


def transform_vehicle_polygons(
    yolo_result,
    homography,
    crop_x,
    crop_y
):
    """
    Transform YOLO bounding boxes from original-image
    coordinates into reference-crop coordinates.
    """

    vehicles = []

    for box in yolo_result.boxes:
        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy()
        )

        confidence = float(box.conf[0].item())

        # Four corners of original bounding box
        corners = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(
            corners, homography
        ).reshape(-1, 2)

        # Convert reference-image coordinates
        # into cropped-image coordinates
        transformed[:, 0] -= crop_x
        transformed[:, 1] -= crop_y

        # Transform the original box center as well
        center_original = np.float32([[
            [(x1 + x2) / 2, (y1 + y2) / 2]
        ]])

        center = cv2.perspectiveTransform(
            center_original, homography
        )[0, 0]

        center = (
            float(center[0] - crop_x),
            float(center[1] - crop_y)
        )

        vehicles.append({
            "polygon": np.round(
                transformed
            ).astype(np.int32),
            "center": center,
            "confidence": confidence
        })

    return vehicles


def classify_polygon_occupancy(
    vehicles,
    parking_spaces,
    image_shape,
    overlap_threshold=0.15
):
    """
    Classify occupancy using transformed vehicle polygons.

    A parking space is occupied when:
    - the vehicle center lies inside the parking polygon
    - overlap exceeds the threshold
    """

    height, width = image_shape[:2]

    predictions = {}

    for space_id, corners in parking_spaces.items():

        space_polygon = np.array(
            corners, dtype=np.int32
        )

        space_mask = np.zeros(
            (height, width), dtype=np.uint8
        )

        cv2.fillPoly(
            space_mask,
            [space_polygon],
            1
        )

        space_area = np.count_nonzero(
            space_mask
        )

        predictions[space_id] = "VACANT"

        if space_area == 0:
            continue

        for vehicle in vehicles:

            # Check vehicle center
            inside = cv2.pointPolygonTest(
                space_polygon,
                vehicle["center"],
                False
            ) >= 0

            if not inside:
                continue

            # Create vehicle polygon mask
            vehicle_mask = np.zeros(
                (height, width), dtype=np.uint8
            )

            cv2.fillPoly(
                vehicle_mask,
                [vehicle["polygon"]],
                1
            )

            intersection = cv2.bitwise_and(
                space_mask,
                vehicle_mask
            )

            overlap = (
                np.count_nonzero(intersection)
                / space_area
            )

            if overlap >= overlap_threshold:
                predictions[space_id] = "OCCUPIED"

                print(
                    f"{space_id}: OCCUPIED "
                    f"| overlap={overlap:.3f} "
                    f"| confidence={vehicle['confidence']:.2f}"
                )

                break

        if predictions[space_id] == "VACANT":
            print(f"{space_id}: VACANT")

    return predictions
