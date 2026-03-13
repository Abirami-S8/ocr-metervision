"""
tests/test_pipeline.py
Basic unit tests — run with: pytest tests/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from preprocessing.quality_check import check_quality
from preprocessing.enhance import auto_enhance
from preprocessing.dewarp import dewarp
from postprocessing.field_parser import validate_field, parse_and_validate
from models.llm_corrector import rule_based_fix, extract_numbers_from_text


def make_test_image(w=640, h=480, brightness=128):
    img = np.full((h, w, 3), brightness, dtype=np.uint8)
    return img


class TestQualityCheck:
    def test_normal_image(self):
        img = make_test_image()
        qr = check_quality(img)
        assert qr.quality_score >= 0.0
        assert isinstance(qr.flags, list)

    def test_dark_image(self):
        img = make_test_image(brightness=20)
        qr = check_quality(img)
        assert qr.is_dark

    def test_bright_image(self):
        img = make_test_image(brightness=240)
        qr = check_quality(img)
        assert qr.is_overexposed


class TestEnhance:
    def test_enhance_returns_same_shape(self):
        img = make_test_image()
        enhanced = auto_enhance(img)
        assert enhanced.shape == img.shape


class TestDewarp:
    def test_dewarp_returns_array(self):
        img = make_test_image()
        result, _ = dewarp(img)
        assert isinstance(result, np.ndarray)


class TestFieldParser:
    def test_validate_kwh_pass(self):
        fr = validate_field("kwh", "002090.3", 0.95,
                             {"kwh": {"pattern": r"^\d{1,7}\.\d{1}$"}})
        assert fr.pass_fail == "PASS"

    def test_validate_kwh_fail_bad_format(self):
        fr = validate_field("kwh", "ABC", 0.9,
                             {"kwh": {"pattern": r"^\d{1,7}\.\d{1}$"}})
        assert fr.pass_fail in ("FAIL", "WARN")

    def test_missing_field(self):
        fr = validate_field("kvah", "", 0.0, {})
        assert fr.pass_fail == "MISSING"


class TestLLMCorrector:
    def test_rule_based_fix(self):
        assert rule_based_fix("OO2O9O.3") == "002090.3"
        assert rule_based_fix("lI1") == "111"

    def test_extract_numbers(self):
        text = "002090.3 kWh | SN TN123456"
        result = extract_numbers_from_text(text)
        assert result["kwh"]["value"] is not None
        assert "002090.3" in result["kwh"]["value"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])