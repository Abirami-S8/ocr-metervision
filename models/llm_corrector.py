"""
models/llm_corrector.py
LLM-based post-correction for OCR outputs.
Uses hackathon-approved small models:
  - Qwen/Qwen2-0.5B-Instruct  (default — fastest, ~1GB RAM)
  - microsoft/phi-2             (better quality, ~3GB RAM)
  - google/gemma-2b-it          (good, ~5GB RAM)
  - LaMini-GPT 1.5B             (offline-friendly)

The LLM is given the raw OCR text + context and asked to:
  1. Identify the 5 meter fields
  2. Correct obvious OCR errors (O→0, l→1, S→5, etc.)
  3. Fix decimal placement
  4. Return structured JSON
"""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_pipeline = None
_loaded_model = None

# Common OCR character confusions for numeric meter displays
OCR_FIXES = {
    "O": "0", "o": "0",
    "l": "1", "I": "1", "i": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
    "q": "9",
    "D": "0",
}

SYSTEM_PROMPT = """You are a smart electricity meter OCR correction assistant.
You receive raw OCR text extracted from a meter photo and must identify and correct 5 fields.
Respond ONLY with valid JSON. No explanation. No markdown.

Fields to extract:
- serial_number: meter serial (alphanumeric, 6-20 chars)
- kwh: active energy reading (format: XXXXXX.X)
- kvah: apparent energy reading (format: XXXXXX.X)  
- md_kw: maximum demand in kW (format: XXXX.XX)
- demand_kva: demand in kVA (format: XXXX.XX)

Rules:
1. Fix OCR errors: O→0, l→1, I→1, S→5, B→8, Z→2, G→6
2. Decimal placement: meter displays typically show 1 decimal for kWh/kVAh
3. If a field is not present, use null
4. Return confidence 0.0-1.0 for each field

Output format (JSON only):
{
  "serial_number": {"value": "ABC123", "confidence": 0.9},
  "kwh": {"value": "002090.3", "confidence": 0.95},
  "kvah": {"value": null, "confidence": 0.0},
  "md_kw": {"value": null, "confidence": 0.0},
  "demand_kva": {"value": null, "confidence": 0.0}
}"""


def _load_pipeline(model_name: str, device: str = "cpu"):
    global _pipeline, _loaded_model
    if _pipeline is None or _loaded_model != model_name:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        logger.info(f"Loading LLM corrector: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(device)
            _pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device if device != "cpu" else -1,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            _loaded_model = model_name
        except Exception as e:
            logger.warning(f"LLM load failed ({e}). Falling back to rule-based correction.")
            _pipeline = None
    return _pipeline


def rule_based_fix(text: str) -> str:
    """Apply simple character substitution rules for numeric fields."""
    result = []
    for ch in text:
        result.append(OCR_FIXES.get(ch, ch))
    return "".join(result)


def extract_numbers_from_text(raw_text: str) -> Dict[str, Any]:
    """
    Rule-based field extraction as fallback when LLM is unavailable.
    Looks for patterns matching meter display values.
    """
    fixed = rule_based_fix(raw_text)

    # Pattern: 6-7 digit number with 1 decimal (kWh/kVAh style)
    energy_pattern = re.compile(r'\b(\d{2,7}[\.,]\d{1,2})\b')
    # Pattern: serial (mix of letters and digits)
    serial_pattern = re.compile(r'\b([A-Z0-9]{6,20})\b')

    energy_matches = energy_pattern.findall(fixed)
    serial_matches = serial_pattern.findall(fixed.upper())

    # Normalize decimal separator
    energy_vals = [m.replace(",", ".") for m in energy_matches]

    result = {
        "serial_number": {"value": serial_matches[0] if serial_matches else None, "confidence": 0.6},
        "kwh":          {"value": energy_vals[0] if len(energy_vals) > 0 else None, "confidence": 0.7},
        "kvah":         {"value": energy_vals[1] if len(energy_vals) > 1 else None, "confidence": 0.65},
        "md_kw":        {"value": energy_vals[2] if len(energy_vals) > 2 else None, "confidence": 0.6},
        "demand_kva":   {"value": energy_vals[3] if len(energy_vals) > 3 else None, "confidence": 0.6},
    }
    return result


def llm_correct(raw_ocr_text: str,
                model_name: str = "Qwen/Qwen2-0.5B-Instruct",
                device: str = "cpu") -> Dict[str, Any]:
    """
    Use LLM to correct and structure OCR output.
    Falls back to rule-based extraction if LLM unavailable.
    
    Returns dict with 5 fields, each having {value, confidence}.
    """
    # Always run rule-based as a baseline
    rule_result = extract_numbers_from_text(raw_ocr_text)

    try:
        pipe = _load_pipeline(model_name, device)
        if pipe is None:
            logger.info("LLM unavailable — using rule-based extraction")
            return rule_result

        prompt = f"{SYSTEM_PROMPT}\n\nRaw OCR text:\n{raw_ocr_text}\n\nJSON output:"

        output = pipe(prompt, max_new_tokens=300, return_full_text=False)
        generated = output[0]["generated_text"].strip()

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', generated, re.DOTALL)
        if not json_match:
            logger.warning("LLM did not return valid JSON, using rule-based")
            return rule_result

        llm_result = json.loads(json_match.group())

        # Merge: use LLM where confident, rule-based as fallback
        merged = {}
        for field in ["serial_number", "kwh", "kvah", "md_kw", "demand_kva"]:
            llm_field = llm_result.get(field, {})
            rule_field = rule_result.get(field, {})

            llm_conf = llm_field.get("confidence", 0) if isinstance(llm_field, dict) else 0
            rule_conf = rule_field.get("confidence", 0)

            if llm_conf >= rule_conf and llm_field.get("value") is not None:
                merged[field] = llm_field
            else:
                merged[field] = rule_field

        return merged

    except Exception as e:
        logger.warning(f"LLM correction failed: {e}. Using rule-based.")
        return rule_result