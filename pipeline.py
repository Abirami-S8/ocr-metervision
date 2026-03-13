"""
pipeline.py
Main OCR pipeline orchestrator.
Ties together: quality check → enhance → dewarp → OCR ensemble → validate
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Union, Dict, Any
import yaml

from preprocessing.quality_check import check_quality, QualityReport
from preprocessing.enhance import auto_enhance
from preprocessing.dewarp import dewarp, detect_display_region
from models.ensemble import ensemble_ocr
from postprocessing.field_parser import parse_and_validate, ExtractionResult

logger = logging.getLogger(__name__)


def _load_cfg():
    cfg_path = Path(__file__).parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def process_image(image_input: Union[str, np.ndarray, bytes],
                   cfg: Dict = None) -> ExtractionResult:
    """
    Full pipeline: image → ExtractionResult with 5 fields + Pass/Fail.
    
    Args:
        image_input: file path (str), numpy array (BGR), or raw bytes
        cfg: optional config dict (loaded from config.yaml if None)
    
    Returns:
        ExtractionResult with all field values, confidences, Pass/Fail
    """
    if cfg is None:
        cfg = _load_cfg()

    t_start = time.time()
    image_path = ""

    # ── Load image ────────────────────────────────────────────────────────────
    if isinstance(image_input, str):
        image_path = image_input
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Cannot load image: {image_input}")
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image bytes")
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    notes = []

    # ── Quality Check ─────────────────────────────────────────────────────────
    qr: QualityReport = check_quality(image, cfg.get("quality_flags"))
    logger.info(f"Quality: score={qr.quality_score:.2f}, flags={qr.flags}")
    if qr.flags:
        notes.append(f"Quality issues: {', '.join(qr.flags)}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    pre_cfg = cfg.get("preprocessing", {})

    # Dewarp / deskew
    if pre_cfg.get("dewarp", {}).get("enabled", True):
        image, was_dewarped = dewarp(image)
        if was_dewarped:
            notes.append("Perspective correction applied")
        else:
            notes.append("Deskew applied (no clean quad found)")

    # Enhance (CLAHE + denoise + sharpen)
    image = auto_enhance(image, brightness_mean=qr.brightness_mean)
    notes.append("Image enhancement applied")

    # Try to isolate display region for a focused OCR pass
    display_crop = detect_display_region(image)
    ocr_image = display_crop if display_crop is not None else image
    if display_crop is not None:
        notes.append("Display region detected and cropped")

    # ── OCR Ensemble ──────────────────────────────────────────────────────────
    ocr_cfg = cfg.get("ocr", {})
    model_cfg = cfg.get("models", {})

    ocr_output = ensemble_ocr(
        ocr_image,
        use_easyocr=True,
        use_trocr=True,
        use_llm=model_cfg.get("llm_corrector", {}).get("enabled", True),
        cfg=cfg
    )

    # If display crop failed to extract enough text, retry on full image
    if all(f.get("value") is None for f in ocr_output["fields"].values()):
        logger.warning("Display crop OCR found nothing — retrying on full image")
        notes.append("Fallback: full-image OCR used")
        ocr_output = ensemble_ocr(image, use_easyocr=True, use_trocr=True, cfg=cfg)

    # ── Validate + Structure ──────────────────────────────────────────────────
    result = parse_and_validate(
        ocr_output["fields"],
        image_path=image_path,
        quality_report=qr
    )
    result.processing_notes = notes

    elapsed = time.time() - t_start
    logger.info(f"Pipeline done in {elapsed:.2f}s | overall_pass={result.overall_pass}")

    return result


def process_batch(input_dir: str,
                   output_dir: str,
                   output_format: str = "json",
                   workers: int = 2) -> Dict[str, Any]:
    """
    Batch process all images in input_dir (recursively).
    Saves results to output_dir as JSON or CSV.
    """
    import json
    import csv
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    cfg = _load_cfg()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() for e in cfg["api"]["allowed_extensions"]}
    image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in exts]

    logger.info(f"Found {len(image_files)} images in {input_dir}")

    results = []
    errors = []

    def process_one(fp: Path):
        try:
            res = process_image(str(fp), cfg=cfg)
            return fp.name, res.to_dict(), None
        except Exception as e:
            logger.error(f"Error processing {fp}: {e}")
            return fp.name, None, str(e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, f): f for f in image_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            name, data, err = fut.result()
            if err:
                errors.append({"file": name, "error": err})
            else:
                data["file"] = name
                results.append(data)

    # Save JSON
    if output_format in ("json", "both"):
        json_path = output_path / "results.json"
        with open(json_path, "w") as f:
            json.dump({"results": results, "errors": errors}, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

    # Save CSV
    if output_format in ("csv", "both"):
        csv_path = output_path / "results.csv"
        if results:
            flat_rows = []
            for r in results:
                row = {"file": r["file"],
                       "overall_pass": r["overall_pass"],
                       "overall_confidence": r["overall_confidence"],
                       "quality_score": r["quality_score"],
                       "quality_flags": "|".join(r.get("quality_flags", []))}
                for fname in ["serial_number", "kwh", "kvah", "md_kw", "demand_kva"]:
                    fd = r["fields"].get(fname, {})
                    row[f"{fname}_value"] = fd.get("value", "")
                    row[f"{fname}_conf"] = fd.get("confidence", 0)
                    row[f"{fname}_status"] = fd.get("pass_fail", "")
                flat_rows.append(row)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
                writer.writeheader()
                writer.writerows(flat_rows)
            logger.info(f"CSV results saved to {csv_path}")

    summary = {
        "total": len(image_files),
        "processed": len(results),
        "errors": len(errors),
        "pass_rate": sum(1 for r in results if r["overall_pass"]) / max(len(results), 1)
    }
    return summary