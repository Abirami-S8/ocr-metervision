"""
preprocessing/dewarp.py
Detect meter display region and apply perspective correction (dewarp).
Strategy:
  1. Edge detection → find large rectangular contours
  2. If a display-like quad is found → perspective warp to frontal view
  3. If no quad found → use Hough-line-based affine deskew as fallback
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def dewarp(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Main entry: try perspective dewarp, fallback to deskew.
    Returns (corrected_image, was_dewarped).
    """
    result, success = _perspective_dewarp(image)
    if success:
        return result, True
    result = _deskew(image)
    return result, False


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _perspective_dewarp(image: np.ndarray,
                         area_ratio_min: float = 0.04,
                         area_ratio_max: float = 0.95) -> Tuple[np.ndarray, bool]:
    h, w = image.shape[:2]
    total_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 120)
    # Dilate edges to close small gaps
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        ratio = area / total_area
        if ratio < area_ratio_min or ratio > area_ratio_max:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = _order_points(pts)
            tl, tr, br, bl = rect

            # Compute destination size
            width = int(max(
                np.linalg.norm(br - bl),
                np.linalg.norm(tr - tl)
            ))
            height = int(max(
                np.linalg.norm(tr - br),
                np.linalg.norm(tl - bl)
            ))

            if width < 50 or height < 30:
                continue

            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (width, height))
            return warped, True

    return image, False


def _deskew(image: np.ndarray) -> np.ndarray:
    """Affine rotation based on dominant Hough line angle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) < 30:
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 1.0:        # negligible tilt
        return image

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def detect_display_region(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to isolate the LCD/display region of the meter.
    Returns the cropped display image, or None if not found.
    """
    # Look for bright rectangular region (LCD backlight)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold for bright region
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]

    best = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        ratio = area / (w * h)
        if ratio < 0.03 or ratio > 0.8:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        # Displays are typically wider than tall
        if 1.5 < aspect < 8.0:
            score = ratio * min(aspect / 3.0, 1.0)
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)
    return image[y1:y2, x1:x2]