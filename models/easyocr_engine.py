"""
models/easyocr_engine.py
EasyOCR wrapper — uses CRAFT text detector + CRNN recognizer.
No Tesseract, no PaddleOCR.
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

_reader = None   # lazy singleton


def _get_reader(gpu: bool = False, languages: list = None):
    global _reader
    if _reader is None:
        try:
            import easyocr
            langs = languages or ["en"]
            logger.info(f"Initialising EasyOCR (gpu={gpu}, langs={langs})")
            _reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        except ImportError:
            raise RuntimeError("easyocr not installed. Run: pip install easyocr")
    return _reader


def run_easyocr(image: np.ndarray,
                gpu: bool = False,
                languages: list = None,
                detail_level: int = 1) -> List[Dict[str, Any]]:
    """
    Run EasyOCR on a BGR or RGB numpy image.
    Returns list of {text, confidence, bbox} dicts.
    
    Args:
        image: numpy array (H, W, C) — BGR or RGB
        gpu: use GPU if available
        languages: list of language codes, default ["en"]
        detail_level: 0=text only, 1=with bbox+conf, 2=full detail
    """
    reader = _get_reader(gpu=gpu, languages=languages)

    # EasyOCR expects RGB
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = image[:, :, ::-1].copy()   # BGR→RGB
    else:
        rgb = image

    results = reader.readtext(rgb, detail=detail_level,
                               paragraph=False,
                               min_size=10,
                               contrast_ths=0.1,
                               adjust_contrast=0.5,
                               text_threshold=0.5,
                               low_text=0.3)

    parsed = []
    for item in results:
        if detail_level == 0:
            parsed.append({"text": item, "confidence": 1.0, "bbox": None})
        else:
            bbox, text, conf = item
            parsed.append({
                "text": text.strip(),
                "confidence": float(conf),
                "bbox": bbox          # list of 4 [x,y] points
            })

    logger.debug(f"EasyOCR found {len(parsed)} text regions")
    return parsed


def run_easyocr_full_page(image: np.ndarray, **kwargs) -> str:
    """Convenience: return all detected text joined as a single string."""
    results = run_easyocr(image, **kwargs)
    return " | ".join(r["text"] for r in results if r["confidence"] > 0.4)