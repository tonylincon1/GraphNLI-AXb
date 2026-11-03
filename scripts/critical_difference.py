#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from graphnli_axb.critical_difference import (
    generate_critical_difference_reports,
    generate_overall_critical_difference_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Friedman/Nemenyi critical-difference reports.")
    parser.add_argument("summary_json")
    parser.add_argument("--out", default=None, help="Output directory; defaults to summary parent/critical_difference.")
    parser.add_argument("--metric", default="binary/mcc")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--overall", action="store_true", help="Aggregate all evaluation splits per seed before Autorank.")
    args = parser.parse_args()
    summary = Path(args.summary_json)
    output = Path(args.out) if args.out else summary.parent / "critical_difference"
    if args.overall:
        report = generate_overall_critical_difference_report(summary, output, args.metric, args.alpha)
        print(report["split"], report["status"], report.get("friedman", {}))
    else:
        reports = generate_critical_difference_reports(summary, output, args.metric, args.alpha)
        for report in reports:
            print(report["split"], report["status"], report.get("friedman", {}))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
