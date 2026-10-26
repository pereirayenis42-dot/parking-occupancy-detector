
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent

# Replace this with the actual problematic output filename
image_path = (
    ROOT / "results" / "images" / "detect_first"
    / "parking_2026-10-07_1807_detect_first.jpg"
)

output_path = (
    ROOT / "results" / "images"
    / "shadow_alignment_debug.jpg"
)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

# Convert to LAB color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

# Enhance local contrast
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced_l = clahe.apply(l)

enhanced_lab = cv2.merge(
    (enhanced_l, a, b)
)

enhanced = cv2.cvtColor(
    enhanced_lab,
    cv2.COLOR_LAB2BGR
)

# Enlarge for easier inspection
enhanced = cv2.resize(
    enhanced,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC
)

cv2.imwrite(
    str(output_path),
    enhanced
)

print("Saved:", output_path)
