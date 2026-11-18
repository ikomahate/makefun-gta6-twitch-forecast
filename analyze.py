"""Summarize the supplied factual histories without network access."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.input_dir.glob("*.in"))]
    records = model.launch_records(cases[0])
    console = [r for r in records if r.meta[0] and not r.meta[1]]
    report = {"source": "Supplied SullyGnome hourly averages; no data fetched", "launches": len(records),
              "console_launches": [r.name for r in console],
              "console_median_ratio_to_hour1": {str(h + 1): statistics.median(math.exp(r.future[h]) for r in console)
                                                 for h in range(1, 24)}, "practice": []}
    for case in cases:
        target = model.target_record(case)
        analogues = sorted(model.analogue_weights(target, model.launch_records(case)), key=lambda pair: -pair[1])
        report["practice"].append({"id": case["id"], "name": target.name, "hour1": target.first,
            "pre12h_peak_over_hour1": math.exp(target.x[0]), "features": target.x,
            "top_analogues": [{"name": r.name, "weight": w} for r, w in analogues[:8]]})
    gta = next(a for a in cases[0]["comparables"] if "target-history" in a.get("unlock", ""))
    points = gta["hourly"]
    recent = [r for r in points if r["from"] >= "2026-11-16T00:00:00Z"]
    report["gta_prelaunch"] = {"points": len(points), "first": points[0]["from"], "last": points[-1]["from"],
        "largest": sorted(points, key=lambda r: r["viewers"], reverse=True)[:5],
        "sample_median": statistics.median(r["viewers"] for r in points),
        "recent_sample_median": statistics.median(r["viewers"] for r in recent),
        "recent_peak": max(r["viewers"] for r in recent),
        "caution": "Sparse sample medians are not time-weighted means. Premiere is an event, not a launch analogue."}
    Path("analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"console_median_ratios": report["console_median_ratio_to_hour1"],
                      "gta_prelaunch": report["gta_prelaunch"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
