
import cv2
import numpy as np


def check_registration_quality(
    homography,
    original_shape,
    reference_shape,
    parking_spaces,
    crop_ratios,
    minimum_coverage=0.90
):
    """
    Check homography validity and parking-space coverage.

    Returns quality results for each parking space.
    """

    original_h, original_w = original_shape[:2]
    reference_h, reference_w = reference_shape[:2]

    results = {}

    # Validate homography
    if (
        homography is None
        or np.asarray(homography).shape != (3, 3)
        or not np.isfinite(homography).all()
    ):
        return {
            space_id: {
                "status": "UNKNOWN",
                "reason": "Invalid homography",
                "coverage": 0.0
            }
            for space_id in parking_spaces
        }

    try:
        inverse_h = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return {
            space_id: {
                "status": "UNKNOWN",
                "reason": "Homography not invertible",
                "coverage": 0.0
            }
            for space_id in parking_spaces
        }

    crop_x = int(reference_w * crop_ratios[0])
    crop_y = int(reference_h * crop_ratios[1])

    # Original-image visibility mask
    original_mask = np.ones(
        (original_h, original_w),
        dtype=np.uint8
    )

    for space_id, corners in parking_spaces.items():
        # Polygon coordinates in full reference image
        polygon = np.asarray(
            corners, dtype=np.float32
        ).copy()

        polygon[:, 0] += crop_x
        polygon[:, 1] += crop_y

        # Project reference space back to original photo
        projected = cv2.perspectiveTransform(
            polygon.reshape(-1, 1, 2),
            inverse_h
        ).reshape(-1, 2)

        if not np.isfinite(projected).all():
            results[space_id] = {
                "status": "UNKNOWN",
                "reason": "Invalid projected coordinates",
                "coverage": 0.0
            }
            continue

        # Measure fraction of polygon covered by original photo
        # by clipping against original-image rectangle.
        projected_polygon = projected.astype(np.float32)

        image_rectangle = np.float32([
            [0, 0],
            [original_w - 1, 0],
            [original_w - 1, original_h - 1],
            [0, original_h - 1]
        ])

        area = abs(cv2.contourArea(projected_polygon))

        if area < 1e-6:
            coverage = 0.0
        else:
            intersection_area, _ = (
                cv2.intersectConvexConvex(
                    projected_polygon,
                    image_rectangle
                )
            )

            coverage = max(
                0.0,
                min(1.0, float(intersection_area) / area)
            )

        if coverage < minimum_coverage:
            results[space_id] = {
                "status": "UNKNOWN",
                "reason": "Parking space outside photo coverage",
                "coverage": coverage
            }
        else:
            results[space_id] = {
                "status": "OK",
                "reason": "Coverage check passed",
                "coverage": coverage
            }

    return results
