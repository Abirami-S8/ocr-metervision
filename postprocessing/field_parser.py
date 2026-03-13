"""
postprocessing/field_parser.py
Validate extracted field values against expected patterns.
Produce Pass/Fail per field + overall result.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field


def _load_cfg():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)["fields"]


FIELD_NAMES = ["serial_number", "kwh", "kvah", "md_kw", "demand_kva"]


@dataclass
class FieldResult:
    field_name: str
    raw_value: str
    normalized_value: str
    confidence: float
    pattern_match: bool
    pass_fail: str          # "PASS" | "FAIL" | "WARN" | "MISSING"
    reason: str = ""


@dataclass
class ExtractionResult:
    image_path: str = ""
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    overall_pass: bool = False
    overall_confidence: float = 0.0
    quality_flags: list = field(default_factory=list)
    quality_score: float = 1.0
    processing_notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "overall_pass": self.overall_pass,
            "overall_confidence": round(self.overall_confidence, 4),
            "quality_score": round(self.quality_score, 4),
            "quality_flags": self.quality_flags,
            "processing_notes": self.processing_notes,
            "fields": {
                name: {
                    "value": fr.normalized_value,
                    "raw_value": fr.raw_value,
                    "confidence": round(fr.confidence, 4),
                    "pass_fail": fr.pass_fail,
                    "reason": fr.reason
                }
                for name, fr in self.fields.items()
            }
        }


def normalize_value(value: str, field_name: str) -> str:
    """Normalize extracted field value."""
    if not value:
        return ""
    v = value.strip().upper()
    # Normalize decimal separator
    v = v.replace(",", ".")
    # Remove spaces within numbers
    if field_name != "serial_number":
        v = re.sub(r'\s+', '', v)
    return v


def validate_field(field_name: str,
                   value: str,
                   confidence: float,
                   cfg: Dict) -> FieldResult:
    """Validate a single field value."""
    field_cfg = cfg.get(field_name, {})
    pattern = field_cfg.get("pattern", "")

    raw = value or ""
    normalized = normalize_value(raw, field_name)

    if not normalized:
        return FieldResult(
            field_name=field_name,
            raw_value=raw,
            normalized_value="",
            confidence=0.0,
            pattern_match=False,
            pass_fail="MISSING",
            reason="No value extracted"
        )

    pattern_match = bool(re.match(pattern, normalized)) if pattern else True
    pass_fail = "PASS" if pattern_match and confidence >= 0.6 else (
        "WARN" if pattern_match and confidence >= 0.4 else "FAIL"
    )
    reason = "" if pass_fail == "PASS" else (
        f"Low confidence ({confidence:.2f})" if pattern_match else
        f"Pattern mismatch (expected {pattern}, got '{normalized}')"
    )

    return FieldResult(
        field_name=field_name,
        raw_value=raw,
        normalized_value=normalized,
        confidence=confidence,
        pattern_match=pattern_match,
        pass_fail=pass_fail,
        reason=reason
    )


def parse_and_validate(extracted_fields: Dict[str, Dict],
                        image_path: str = "",
                        quality_report=None) -> ExtractionResult:
    """
    Given the LLM/OCR extracted fields dict, validate all 5 fields
    and return a complete ExtractionResult.
    """
    cfg = _load_cfg()
    result = ExtractionResult(image_path=image_path)

    if quality_report:
        result.quality_flags = quality_report.flags
        result.quality_score = quality_report.quality_score

    field_results = {}
    confidences = []

    for fname in FIELD_NAMES:
        fdata = extracted_fields.get(fname, {})
        value = fdata.get("value") if isinstance(fdata, dict) else fdata
        conf = fdata.get("confidence", 0.0) if isinstance(fdata, dict) else 0.5

        fr = validate_field(fname, str(value) if value else "", float(conf), cfg)
        field_results[fname] = fr
        confidences.append(fr.confidence)

    result.fields = field_results
    result.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Overall pass: all non-MISSING fields must PASS (or WARN)
    statuses = [fr.pass_fail for fr in field_results.values() if fr.pass_fail != "MISSING"]
    result.overall_pass = all(s in ("PASS", "WARN") for s in statuses) and len(statuses) > 0

    return result