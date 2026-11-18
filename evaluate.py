"""Offline evaluation. Official practice labels are optional and NEVER imported by main.

python evaluate.py --input-dir PATH [--answers-dir PATH] [--output-dir out]
Answers, if available, are <id>.json with {"hours": [{"hour": 2, "viewers": ...}]}.
Without them we measure leave-one-launch-out backtests on supplied comparables,
and validate practice forecasts without inventing their scores.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model


def baseline(first):
    return {h: [math.log(math.floor(max(1, first) * math.exp(z * 0.15 * math.sqrt(h)) * 100 + 0.5) / 100)
                for z in (-1.645, -0.674, 0, 0.674, 1.645)] for h in range(1, 24)}


def metrics(predictions, answers):
    losses, errors, covered = [], [], []
    for h, actual in answers.items():
        if h not in predictions or actual is None:
            continue
        y = model.log_count(actual)
        row = predictions[h]
        for q, level in zip(row, model.LEVELS):
            delta = y - q
            losses.append(2 * max(level * delta, (level - 1) * delta))
        errors.append(abs(y - row[2]))
        covered.append(float(row[0] <= y <= row[4]))
    if not losses:
        return {"hours": 0, "loss": None}
    return {"hours": len(errors), "loss": statistics.mean(losses),
            "typical_miss_pct": 100 * math.expm1(statistics.mean(errors)),
            "coverage90": statistics.mean(covered)}


def validate(output):
    assert set(output) == {"hours"}
    assert len(output["hours"]) == 23
    assert sorted(r["hour"] for r in output["hours"]) == list(range(2, 25))
    for row in output["hours"]:
        assert set(row) == {"hour", "q"}
        assert len(row["q"]) == 5
        assert all(type(x) in (int, float) and math.isfinite(x) and 0 < x <= 1e9 for x in row["q"])
        assert row["q"] == sorted(row["q"])
    assert len(json.dumps(output).encode("utf-8")) <= 65536


def summarize(rows):
    total = sum(row["model"]["hours"] for row in rows)
    result = {}
    for method in ("model", "baseline"):
        result[method] = {key: sum(row[method][key] * row[method]["hours"] for row in rows) / total
                          for key in ("loss", "coverage90")}
        result[method]["typical_miss_pct"] = 100 * math.expm1(sum(
            math.log1p(row[method]["typical_miss_pct"] / 100) * row[method]["hours"]
            for row in rows) / total)
    result["launches"] = len(rows)
    result["hours"] = total
    return result


def backtest(records, bandwidth=1.0):
    rows = []
    for target in records:
        training = [r for r in records if r.name != target.name]
        predicted = model.predict_logs(target, training, bandwidth)
        answers = {h: math.exp(v) * target.first for h, v in target.future.items()}
        rows.append({"name": target.name, "console": bool(target.meta[0]),
                     "early_access": bool(target.meta[1]), "first": target.first,
                     "model": metrics(predicted, answers), "baseline": metrics(baseline(target.first), answers)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--answers-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--report", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--bandwidth", type=float, default=1.0)
    args = parser.parse_args()
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.input_dir.glob("*.in"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"python": sys.version, "practice": [], "label_policy":
              "Practice answers are evaluation-only. Held-out launch is removed before fitting."}
    for case in cases:
        started = time.perf_counter()
        output = model.forecast_case(case)
        elapsed = time.perf_counter() - started
        validate(output)
        encoded = json.dumps(output, allow_nan=False, separators=(",", ":")) + "\n"
        (args.output_dir / (case["id"] + ".out")).write_text(encoded, encoding="utf-8")
        item = {"id": case["id"], "name": case["game"]["name"], "seconds": elapsed,
                "valid": True, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        answer_file = args.answers_dir / (case["id"] + ".json") if args.answers_dir else None
        if answer_file and answer_file.is_file():
            labels = json.loads(answer_file.read_text(encoding="utf-8"))
            answers = {row["hour"] - 1: row["viewers"] for row in labels["hours"] if 2 <= row["hour"] <= 24}
            logs = {r["hour"] - 1: [math.log(x) for x in r["q"]] for r in output["hours"]}
            item["model"] = metrics(logs, answers)
            item["baseline"] = metrics(baseline(model.target_record(case).first), answers)
        else:
            item["score_status"] = "unavailable: official future labels are not in the kit"
        report["practice"].append(item)
    records = model.launch_records(cases[0])
    started = time.perf_counter()
    rows = backtest(records, args.bandwidth)
    report["backtest_seconds"] = time.perf_counter() - started
    report["backtest"] = rows
    report["all_comparables"] = summarize(rows)
    selected = [r for r in rows if r["console"] and not r["early_access"]]
    report["console_without_early_access"] = summarize(selected)
    scored = [row for row in report["practice"] if "model" in row]
    if scored:
        report["scored_practice"] = summarize(scored)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_comparables", "console_without_early_access", "backtest_seconds")}, indent=2))
    for row in rows:
        if row["console"]:
            print(row["name"], round(row["model"]["loss"], 6), round(row["baseline"]["loss"], 6))


if __name__ == "__main__":
    main()
