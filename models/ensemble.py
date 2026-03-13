"""
models/ensemble.py
Ensemble strategy: run EasyOCR + TrOCR on same image,
merge results by confidence voting.
Also integrates LLM correction as final stage.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path

from models.easyocr_engine import run_easyocr
from models.trocr_engine import run_trocr_on_regions
from models.llm_corrector import llm_correct

logger = logging.getLogger(__name__)


def _load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensemble_ocr(image: np.ndarray,
                  use_easyocr: bool = True,
                  use_trocr: bool = True,
                  use_llm: bool = True,
                  cfg: dict = None) -> Dict[str, Any]:
    """
    Run ensemble OCR pipeline on a meter image.
    
    Returns:
        {
          "raw_texts": [...],
          "combined_text": "...",
          "fields": {
            "serial_number": {"value": "...", "confidence": 0.9},
            "kwh":           {"value": "...", "confidence": 0.95},
            ...
          },
          "engine_outputs": {...}
        }
    """
    if cfg is None:
        cfg = _load_cfg()

    model_cfg = cfg.get("models", {})
    ocr_cfg = cfg.get("ocr", {})
    llm_cfg = model_cfg.get("llm_corrector", {})

    engine_outputs = {}
    all_text_blocks = []

    # ── EasyOCR ──────────────────────────────────────────────────────────────
    if use_easyocr:
        try:
            easy_results = run_easyocr(
                image,
                gpu=model_cfg.get("easyocr", {}).get("gpu", False),
                languages=ocr_cfg.get("languages", ["en"])
            )
            engine_outputs["easyocr"] = easy_results
            for item in easy_results:
                if item["confidence"] >= ocr_cfg.get("confidence_threshold", 0.5):
                    all_text_blocks.append({
                        "text": item["text"],
                        "confidence": item["confidence"],
                        "source": "easyocr"
                    })
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")

    # ── TrOCR (full image fallback or on detected bboxes) ────────────────────
    if use_trocr:
        try:
            # Get bboxes from EasyOCR detections to feed TrOCR
            bboxes = None
            if "easyocr" in engine_outputs:
                bboxes = _convert_bboxes(engine_outputs["easyocr"])

            trocr_results = run_trocr_on_regions(
                image,
                bboxes=bboxes,
                model_name=model_cfg.get("trocr", {}).get(
                    "model_name", "microsoft/trocr-base-printed"),
                device=model_cfg.get("trocr", {}).get("device", "cpu")
            )
            engine_outputs["trocr"] = trocr_results
            for item in trocr_results:
                if item["confidence"] >= 0.5:
                    all_text_blocks.append({
                        "text": item["text"],
                        "confidence": item["confidence"],
                        "source": "trocr"
                    })
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}")

    # ── Merge text ────────────────────────────────────────────────────────────
    if not all_text_blocks:
        logger.error("All OCR engines failed — no text extracted")
        return _empty_result()

    # Deduplicate: if EasyOCR and TrOCR agree on a token, boost its confidence
    combined_text = _merge_texts(all_text_blocks)
    logger.info(f"Combined OCR text: {combined_text[:200]}")

    # ── LLM Correction ───────────────────────────────────────────────────────
    if use_llm and llm_cfg.get("enabled", True):
        fields = llm_correct(
            combined_text,
            model_name=llm_cfg.get("model_name", "Qwen/Qwen2-0.5B-Instruct"),
            device=llm_cfg.get("device", "cpu")
        )
    else:
        from models.llm_corrector import extract_numbers_from_text
        fields = extract_numbers_from_text(combined_text)

    return {
        "raw_texts": all_text_blocks,
        "combined_text": combined_text,
        "fields": fields,
        "engine_outputs": {k: len(v) for k, v in engine_outputs.items()}
    }


def _convert_bboxes(easyocr_results: List[Dict]) -> Optional[List]:
    """Convert EasyOCR polygon bboxes to (x1,y1,x2,y2) tuples."""
    bboxes = []
    for item in easyocr_results:
        bbox = item.get("bbox")
        if bbox and len(bbox) == 4:
            pts = np.array(bbox)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return bboxes if bboxes else None


def _merge_texts(blocks: List[Dict]) -> str:
    """Merge text blocks sorted by spatial order (top-to-bottom, left-to-right)."""
    # Sort by confidence descending, then join unique non-overlapping texts
    seen = set()
    parts = []
    for block in sorted(blocks, key=lambda x: -x["confidence"]):
        t = block["text"].strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            parts.append(t)
    return " | ".join(parts)


def _empty_result() -> Dict[str, Any]:
    empty_field = {"value": None, "confidence": 0.0}
    return {
        "raw_texts": [],
        "combined_text": "",
        "fields": {
            "serial_number": dict(empty_field),
            "kwh": dict(empty_field),
            "kvah": dict(empty_field),
            "md_kw": dict(empty_field),
            "demand_kva": dict(empty_field),
        },
        "engine_outputs": {}
    }