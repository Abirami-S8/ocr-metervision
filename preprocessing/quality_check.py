"""
preprocessing/quality_check.py
Assess image quality before OCR: blur, brightness, tilt, resolution.
Returns quality flags and a composite quality score.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List
import yaml
from pathlib import Path


def _load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)["quality_flags"]


@dataclass
class QualityReport:
    is_blurry: bool = False
    is_dark: bool = False
    is_overexposed: bool = False
    is_tilted: bool = False
    low_resolution: bool = False
    blur_score: float = 0.0          # higher = sharper
    brightness_mean: float = 0.0
    estimated_tilt_deg: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    flags: List[str] = field(default_factory=list)
    quality_score: float = 1.0       # 0.0 (bad) – 1.0 (perfect)
    pass_qc: bool = True


def compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — industry standard blur metric."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def estimate_tilt(gray: np.ndarray) -> float:
    """
    Use Hough line transform to estimate dominant text-line angle.
    Returns degrees off-horizontal (0 = perfectly level).
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = np.degrees(theta) - 90
        if abs(angle_deg) < 45:          # ignore near-vertical lines
            angles.append(angle_deg)

    if not angles:
        return 0.0
    return float(np.median(angles))


def check_quality(image: np.ndarray, cfg: dict = None) -> QualityReport:
    """
    Run all quality checks on a BGR image (as loaded by cv2.imread).
    Returns a QualityReport.
    """
    if cfg is None:
        cfg = _load_cfg()

    report = QualityReport()
    h, w = image.shape[:2]
    report.resolution = (w, h)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Blur ---
    blur = compute_blur_score(gray)
    report.blur_score = blur
    if blur < cfg["blur_threshold"]:
        report.is_blurry = True
        report.flags.append(f"BLUR (score={blur:.1f} < {cfg['blur_threshold']})")

    # --- Brightness ---
    brightness = compute_brightness(gray)
    report.brightness_mean = brightness
    if brightness < cfg["brightness_low"]:
        report.is_dark = True
        report.flags.append(f"DARK (mean={brightness:.1f})")
    elif brightness > cfg["brightness_high"]:
        report.is_overexposed = True
        report.flags.append(f"OVEREXPOSED (mean={brightness:.1f})")

    # --- Tilt ---
    tilt = estimate_tilt(gray)
    report.estimated_tilt_deg = tilt
    if abs(tilt) > cfg["tilt_max_deg"]:
        report.is_tilted = True
        report.flags.append(f"TILT ({tilt:.1f}°)")

    # --- Resolution ---
    if w < 400 or h < 300:
        report.low_resolution = True
        report.flags.append(f"LOW_RES ({w}x{h})")

    # --- Composite score (0–1) ---
    penalties = 0.0
    if report.is_blurry:       penalties += 0.35
    if report.is_dark:         penalties += 0.20
    if report.is_overexposed:  penalties += 0.15
    if report.is_tilted:       penalties += 0.15
    if report.low_resolution:  penalties += 0.15
    report.quality_score = max(0.0, 1.0 - penalties)
    report.pass_qc = len(report.flags) == 0

    return report