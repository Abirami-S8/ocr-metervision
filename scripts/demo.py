#!/usr/bin/env python3
"""
scripts/demo.py
Quick demo: run OCR on the sample meter image provided.
Requires: pip install easyocr transformers torch opencv-python-headless Pillow

Usage:
  python scripts/demo.py --image path/to/meter.jpg
"""

import argparse
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Smart Meter OCR Demo")
    parser.add_argument("--image", "-i", required=True, help="Path to meter image")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM correction layer")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("  Smart Meter OCR — Demo")
    print(f"{'='*55}")
    print(f"  Image: {args.image}")
    print()

    # Patch config to skip LLM if requested
    if args.no_llm:
        import yaml
        cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["models"]["llm_corrector"]["enabled"] = False
    else:
        cfg = None

    from pipeline import process_image
    result = process_image(args.image, cfg=cfg)
    rd = result.to_dict()

    # Print results
    status = "✅ PASS" if rd["overall_pass"] else "❌ FAIL"
    print(f"  Overall: {status}")
    print(f"  Confidence: {rd['overall_confidence']:.1%}")
    print(f"  Quality:    {rd['quality_score']:.1%}")
    if rd["quality_flags"]:
        print(f"  Flags: {', '.join(rd['quality_flags'])}")
    print()
    print(f"  {'Field':<20} {'Value':<15} {'Confidence':<12} {'Status'}")
    print(f"  {'-'*60}")
    for fname, fdata in rd["fields"].items():
        val = fdata.get("value") or "—"
        conf = f"{fdata.get('confidence', 0):.0%}"
        status = fdata.get("pass_fail", "MISSING")
        print(f"  {fname:<20} {val:<15} {conf:<12} {status}")

    print(f"\n  Processing notes:")
    for note in rd.get("processing_notes", []):
        print(f"    • {note}")

    print(f"\n  Full JSON:")
    print(json.dumps(rd, indent=2))
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()