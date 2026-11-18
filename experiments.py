"""Small, reproducible development sweep using only comparable-launch labels.

Nested validation removes each outer launch from the entire inner selection
procedure. Its future is accessed only by the final scorer for that fold.
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model
from evaluate import baseline, metrics, summarize


WEIGHT_CONFIGS = [{"bandwidth": b, "ridge": r, "residual_scale": 1.0}
                  for b in (0.55, 0.75, 1.0) for r in (0.06, 0.2, 0.6)]
CALIBRATION_CONFIGS = [{"bandwidth": 1.0, "ridge": r, "residual_scale": s}
                       for r in (0.1, 0.3) for s in (0.5, 0.75, 1.0)]


def score(target, training, config):
    predictions = model.predict_logs(target, training, **config)
    answers = {h: target.first * math.exp(v) for h, v in target.future.items()}
    return metrics(predictions, answers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--nested", action="store_true")
    parser.add_argument("--stage", choices=("weights", "calibration"), default="calibration")
    args = parser.parse_args()
    configs = WEIGHT_CONFIGS if args.stage == "weights" else CALIBRATION_CONFIGS
    case = json.loads(next(args.input_dir.glob("*.in")).read_text(encoding="utf-8"))
    records = model.launch_records(case)
    selected = [r for r in records if r.meta[0] and not r.meta[1]]
    sweeps = []
    for config in configs:
        scores = [score(target, [r for r in records if r.name != target.name], config) for target in selected]
        row = {"config": config, "mean_loss": sum(s["loss"] for s in scores) / len(scores),
               "coverage90": sum(s["coverage90"] for s in scores) / len(scores)}
        sweeps.append(row)
        print(json.dumps(row), flush=True)
    report = {"sweep": sweeps, "selection_data": "7 non-practice console launches, without early access"}
    if args.nested:
        outer_rows = []
        for outer in selected:
            available = [r for r in records if r.name != outer.name]
            inner = [r for r in selected if r.name != outer.name]
            losses = []
            for config in configs:
                loss = sum(score(t, [r for r in available if r.name != t.name], config)["loss"]
                           for t in inner) / len(inner)
                losses.append(loss)
            config = configs[min(range(len(losses)), key=lambda i: losses[i])]
            row = {"name": outer.name, "selected": config, "model": score(outer, available, config),
                   "baseline": metrics(baseline(outer.first),
                                       {h: outer.first * math.exp(v) for h, v in outer.future.items()})}
            outer_rows.append(row)
            print(json.dumps(row), flush=True)
        report["nested_outer"] = outer_rows
        report["nested_summary"] = summarize(outer_rows)
    Path("experiments-" + args.stage + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
