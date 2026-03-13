"""
api/schemas.py
Pydantic schemas for request / response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class FieldDetail(BaseModel):
    value: Optional[str] = None
    raw_value: Optional[str] = None
    confidence: float = 0.0
    pass_fail: str = "MISSING"
    reason: str = ""


class ExtractionResponse(BaseModel):
    image_path: str = ""
    overall_pass: bool
    overall_confidence: float
    quality_score: float
    quality_flags: List[str] = []
    processing_notes: List[str] = []
    fields: Dict[str, FieldDetail]

    class Config:
        json_schema_extra = {
            "example": {
                "image_path": "meter_001.jpg",
                "overall_pass": True,
                "overall_confidence": 0.91,
                "quality_score": 0.85,
                "quality_flags": [],
                "processing_notes": ["Perspective correction applied"],
                "fields": {
                    "serial_number": {"value": "TN123456", "confidence": 0.88, "pass_fail": "PASS"},
                    "kwh": {"value": "002090.3", "confidence": 0.96, "pass_fail": "PASS"},
                    "kvah": {"value": "002500.1", "confidence": 0.94, "pass_fail": "PASS"},
                    "md_kw": {"value": "15.50", "confidence": 0.89, "pass_fail": "PASS"},
                    "demand_kva": {"value": "18.20", "confidence": 0.87, "pass_fail": "PASS"},
                }
            }
        }


class BatchRequest(BaseModel):
    input_dir: str = Field(default="data/raw_images", description="Path to folder with meter images")
    output_dir: str = Field(default="data/processed", description="Where to save results")
    output_format: str = Field(default="both", description="json | csv | both")
    workers: int = Field(default=2, ge=1, le=16)


class HealthResponse(BaseModel):
    status: str
    version: str