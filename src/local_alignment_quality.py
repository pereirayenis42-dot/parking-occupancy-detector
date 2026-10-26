
import cv2
import numpy as np


def evaluate_local_alignment(
    reference,
    test,
    homography,
    parking_spaces,
    crop_ratios,
    minimum_matches=5,
    maximum_median_error=5.0,
    search_margin=100
):
    """
    Evaluate local alignment using ORB feature matches.

    Returns per-space diagnostics.
    """

    ref_h, ref_w = reference.shape[:2]

    crop_x = int(ref_w * crop_ratios[0])
    crop_y = int(ref_h * crop_ratios[1])

    gray_ref = cv2.cvtColor(
        reference, cv2.COLOR_BGR2GRAY
    )

    gray_test = cv2.cvtColor(
        test, cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(nfeatures=5000)

    kp_ref, des_ref = orb.detectAndCompute(
        gray_ref, None
    )

    kp_test, des_test = orb.detectAndCompute(
        gray_test, None
    )

    results = {}

    if des_ref is None or des_test is None:
        return {
            space_id: {
                "status": "INSUFFICIENT",
                "matches": 0,
                "median_error": None
            }
            for space_id in parking_spaces
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        des_test, des_ref, k=2
    )

    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    if not good:
        return {
            space_id: {
                "status": "INSUFFICIENT",
                "matches": 0,
                "median_error": None
            }
            for space_id in parking_spaces
        }

    test_points = np.float32([
        kp_test[m.queryIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    ref_points = np.float32([
        kp_ref[m.trainIdx].pt
        for m in good
    ]).reshape(-1, 2)

    projected = cv2.perspectiveTransform(
        test_points, homography
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - ref_points,
        axis=1
    )

    for space_id, corners in parking_spaces.items():

        polygon = np.asarray(
            corners, dtype=np.float32
        ).copy()

        polygon[:, 0] += crop_x
        polygon[:, 1] += crop_y

        min_x = polygon[:, 0].min() - search_margin
        max_x = polygon[:, 0].max() + search_margin
        min_y = polygon[:, 1].min() - search_margin
        max_y = polygon[:, 1].max() + search_margin

        # Select matches whose reference locations
        # fall near the parking space.
        nearby = (
            (ref_points[:, 0] >= min_x) &
            (ref_points[:, 0] <= max_x) &
            (ref_points[:, 1] >= min_y) &
            (ref_points[:, 1] <= max_y)
        )

        local_errors = errors[nearby]
        count = len(local_errors)

        if count < minimum_matches:
            status = "INSUFFICIENT"
            median_error = None
        else:
            median_error = float(
                np.median(local_errors)
            )

            status = (
                "PASS"
                if median_error <= maximum_median_error
                else "FAIL"
            )

        results[space_id] = {
            "status": status,
            "matches": count,
            "median_error": median_error
        }

    return results
