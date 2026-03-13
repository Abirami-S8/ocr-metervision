#!/usr/bin/env python3
"""
scripts/evaluate.py
Evaluate OCR accuracy on a labeled benchmark set.

Benchmark CSV format:
  filename,serial_number,kwh,kvah,md_kw,demand_kva

Usage:
  python scripts/evaluate.py --benchmark data/benchmark.csv --images data/benchmark_images/
"""

import argparse
import csv
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

import editdistance
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import process_image

logging.basicConfig(level=logging.WARNING)

FIELDS = ["serial_number", "kwh", "kvah", "md_kw", "demand_kva"]


def exact_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    return pred.strip().upper() == gt.strip().upper()


def char_accuracy(pred: str, gt: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    dist = editdistance.eval(pred.strip(), gt.strip())
    return max(0.0, 1.0 - dist / max(len(gt), 1))


def evaluate(benchmark_csv: str, images_dir: str) -> Dict:
    with open(benchmark_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Evaluating {len(rows)} benchmark images...")

    field_exact = {f: [] for f in FIELDS}
    field_char_acc = {f: [] for f in FIELDS}
    field_conf_when_correct = {f: [] for f in FIELDS}
    field_conf_when_wrong = {f: [] for f in FIELDS}

    for row in rows:
        img_path = str(Path(images_dir) / row["filename"])
        try:
            result = process_image(img_path)
            rd = result.to_dict()
        except Exception as e:
            print(f"  ERROR {row['filename']}: {e}")
            continue

        for f in FIELDS:
            gt = row.get(f, "").strip()
            if not gt:
                continue
            pred_data = rd["fields"].get(f, {})
            pred = (pred_data.get("value") or "").strip()
            conf = pred_data.get("confidence", 0.0)

            em = exact_match(pred, gt)
            ca = char_accuracy(pred, gt)

            field_exact[f].append(int(em))
            field_char_acc[f].append(ca)
            if em:
                field_conf_when_correct[f].append(conf)
            else:
                field_conf_when_wrong[f].append(conf)

    # Compile metrics
    metrics = {}
    for f in FIELDS:
        em_list = field_exact[f]
        ca_list = field_char_acc[f]
        cc = field_conf_when_correct[f]
        cw = field_conf_when_wrong[f]

        metrics[f] = {
            "exact_accuracy": np.mean(em_list) if em_list else 0.0,
            "char_accuracy":  np.mean(ca_list) if ca_list else 0.0,
            "samples":        len(em_list),
            "avg_conf_correct": np.mean(cc) if cc else 0.0,
            "avg_conf_wrong":   np.mean(cw) if cw else 0.0,
        }

    overall_exact = np.mean([v["exact_accuracy"] for v in metrics.values() if v["samples"] > 0])
    overall_char  = np.mean([v["char_accuracy"]  for v in metrics.values() if v["samples"] > 0])

    report = {
        "overall_exact_accuracy": float(overall_exact),
        "overall_char_accuracy":  float(overall_char),
        "per_field": metrics
    }

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Overall Exact Accuracy: {overall_exact:.1%}")
    print(f"  Overall Char Accuracy:  {overall_char:.1%}")
    print(f"{'='*55}")
    for f, m in metrics.items():
        print(f"  {f:<20} exact={m['exact_accuracy']:.1%}  char={m['char_accuracy']:.1%}  n={m['samples']}")
    print(f"{'='*55}\n")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "-b", required=True, help="Benchmark CSV")
    parser.add_argument("--images", "-i", required=True, help="Benchmark images dir")
    parser.add_argument("--output", "-o", default="evaluation_report.json")
    args = parser.parse_args()

    report = evaluate(args.benchmark, args.images)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()