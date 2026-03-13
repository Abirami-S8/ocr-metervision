"""
preprocessing/enhance.py
Image enhancement: CLAHE, denoising, sharpening, night-mode boost.
"""

import cv2
import numpy as np
from typing import Optional


def auto_enhance(image: np.ndarray, brightness_mean: float = None) -> np.ndarray:
    """
    Full auto-enhancement pipeline for meter images.
    Applies: CLAHE → denoise → sharpen → (night boost if needed).
    Input/output: BGR uint8 numpy array.
    """
    if brightness_mean is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(gray))

    img = image.copy()

    # Night mode: gamma correction before other steps
    if brightness_mean < 60:
        img = _gamma_correction(img, gamma=1.8)

    # CLAHE on L channel of LAB
    img = _clahe_lab(img)

    # Denoise (Non-Local Means — better than Gaussian for text)
    img = _nlm_denoise(img)

    # Unsharp mask
    img = _unsharp_mask(img)

    return img


def _gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def _clahe_lab(image: np.ndarray, clip_limit: float = 3.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _nlm_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, h=6, hColor=6,
                                            templateWindowSize=7, searchWindowSize=21)


def _unsharp_mask(image: np.ndarray, strength: float = 1.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
    return cv2.addWeighted(image, strength, blurred, -(strength - 1), 0)