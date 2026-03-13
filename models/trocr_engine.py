"""
models/trocr_engine.py
TrOCR — Microsoft's Vision Encoder–Decoder for printed text OCR.
Model: microsoft/trocr-base-printed  (or trocr-large-printed)
Much better than Tesseract on degraded/low-contrast meter displays.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

_processor = None
_model = None
_device = "cpu"


def _load_model(model_name: str = "microsoft/trocr-base-printed",
                device: str = "cpu"):
    global _processor, _model, _device
    if _model is None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            logger.info(f"Loading TrOCR model: {model_name} on {device}")
            _processor = TrOCRProcessor.from_pretrained(model_name)
            _model = VisionEncoderDecoderModel.from_pretrained(model_name)
            _device = device
            _model = _model.to(device)
            _model.eval()
        except ImportError:
            raise RuntimeError("transformers not installed. Run: pip install transformers")
    return _processor, _model


def run_trocr_on_crop(crop: np.ndarray,
                       model_name: str = "microsoft/trocr-base-printed",
                       device: str = "cpu") -> Dict[str, Any]:
    """
    Run TrOCR on a single cropped text-line image.
    Returns {text, confidence}.
    
    Args:
        crop: numpy BGR or RGB image of a single text line/region
    """
    import torch

    processor, model = _load_model(model_name, device)

    # Convert to PIL RGB
    if isinstance(crop, np.ndarray):
        if crop.ndim == 3 and crop.shape[2] == 3:
            pil_img = Image.fromarray(crop[:, :, ::-1])  # BGR→RGB
        else:
            pil_img = Image.fromarray(crop).convert("RGB")
    else:
        pil_img = crop.convert("RGB")

    # Resize to reasonable height if too small
    w, h = pil_img.size
    if h < 32:
        scale = 32 / h
        pil_img = pil_img.resize((int(w * scale), 32), Image.BICUBIC)

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(pixel_values,
                                  output_scores=True,
                                  return_dict_in_generate=True,
                                  max_new_tokens=32)

    generated_ids = outputs.sequences
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Approximate confidence from token scores
    confidence = 0.85  # TrOCR doesn't expose easy per-output confidence
    if hasattr(outputs, "scores") and outputs.scores:
        import torch.nn.functional as F
        token_probs = [F.softmax(s, dim=-1).max().item() for s in outputs.scores]
        if token_probs:
            confidence = float(np.mean(token_probs))

    return {"text": text, "confidence": confidence}


def run_trocr_on_regions(image: np.ndarray,
                          bboxes: Optional[List] = None,
                          **kwargs) -> List[Dict[str, Any]]:
    """
    Run TrOCR on multiple bounding-box regions of an image.
    If bboxes is None, runs on the full image as one crop.
    
    bboxes: list of (x1, y1, x2, y2) tuples
    """
    results = []
    if bboxes is None:
        result = run_trocr_on_crop(image, **kwargs)
        result["bbox"] = None
        results.append(result)
        return results

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = run_trocr_on_crop(crop, **kwargs)
        result["bbox"] = bbox
        results.append(result)

    return results