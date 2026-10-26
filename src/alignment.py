
import cv2
import numpy as np


def align_images(reference_image, test_image):
    """
    Align test_image to reference_image.

    Both inputs are OpenCV BGR images.

    Returns:
        aligned_image, diagnostics
    """

    ref_gray = cv2.cvtColor(
        reference_image, cv2.COLOR_BGR2GRAY
    )
    test_gray = cv2.cvtColor(
        test_image, cv2.COLOR_BGR2GRAY
    )

    # Detect visual features
    orb = cv2.ORB_create(nfeatures=4000)

    kp_ref, des_ref = orb.detectAndCompute(
        ref_gray, None
    )
    kp_test, des_test = orb.detectAndCompute(
        test_gray, None
    )

    if des_ref is None or des_test is None:
        raise ValueError(
            "Not enough image features for alignment."
        )

    # Match features between photographs
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    pairs = matcher.knnMatch(
        des_test, des_ref, k=2
    )

    # Lowe's ratio test
    good_matches = []

    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 12:
        raise ValueError(
            f"Only {len(good_matches)} good matches."
        )

    # Test-photo points
    src_points = np.float32([
        kp_test[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # Reference-photo points
    dst_points = np.float32([
        kp_ref[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    # Estimate geometric transformation
    homography, inlier_mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        4.0
    )

    if homography is None or inlier_mask is None:
        raise ValueError(
            "Could not estimate a valid homography."
        )

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good_matches)

    # Require enough consistent matches
    if inliers < 10 or inlier_ratio < 0.30:
        raise ValueError(
            f"Unreliable alignment: "
            f"{inliers} inliers, "
            f"ratio={inlier_ratio:.2f}"
        )

    height, width = reference_image.shape[:2]

    aligned = cv2.warpPerspective(
        test_image,
        homography,
        (width, height)
    )

    diagnostics = {
        "reference_features": len(kp_ref),
        "test_features": len(kp_test),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 3)
    }

    return aligned, diagnostics, homography
