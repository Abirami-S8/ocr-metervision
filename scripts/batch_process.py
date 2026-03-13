#!/usr/bin/env python3
"""
scripts/batch_process.py
CLI script for batch processing all meter images in a directory.

Usage:
  python scripts/batch_process.py --input data/raw_images --output data/processed
  python scripts/batch_process.py --input data/raw_images --output data/processed --format both --workers 4
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import process_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


def main():
    parser = argparse.ArgumentParser(description="Smart Meter OCR Batch Processor")
    parser.add_argument("--input", "-i", default="data/raw_images",
                        help="Input directory with meter photos")
    parser.add_argument("--output", "-o", default="data/processed",
                        help="Output directory for results")
    parser.add_argument("--format", "-f", default="both",
                        choices=["json", "csv", "both"],
                        help="Output format")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Parallel workers")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("  Smart Meter OCR — Batch Processor")
    print(f"{'='*50}")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print(f"  Format:  {args.format}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*50}\n")

    summary = process_batch(
        args.input,
        args.output,
        output_format=args.format,
        workers=args.workers
    )

    print(f"\n{'='*50}")
    print("  BATCH COMPLETE")
    print(f"{'='*50}")
    print(f"  Total images:  {summary['total']}")
    print(f"  Processed:     {summary['processed']}")
    print(f"  Errors:        {summary['errors']}")
    print(f"  Pass rate:     {summary['pass_rate']:.1%}")
    print(f"{'='*50}\n")
    print(f"  Results saved to: {args.output}/")


if __name__ == "__main__":
    main()