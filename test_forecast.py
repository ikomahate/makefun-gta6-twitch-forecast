"""Contract, leakage, and execution tests; run with --input-dir KIT/input."""

import argparse
import ast
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main as model
from evaluate import metrics, validate


ROOT = Path(__file__).resolve().parent
CASES = []


class ForecastTests(unittest.TestCase):
    def test_scoring_rule_worked_example(self):
        row = [math.log(x) for x in (600000, 800000, 900000, 1100000, 1500000)]
        result = metrics({1: row, 2: row}, {1: 1000000, 2: 700000})
        self.assertAlmostEqual(result["loss"], 0.112545, places=6)

    def test_target_future_does_not_change_predictions(self):
        case = CASES[0]
        target = model.target_record(case)
        records = model.launch_records(case)
        expected = model.predict_logs(target, records)
        # The evaluator retains labels on its target object, but inference must
        # not inspect them. Poisoning them must have no effect whatsoever.
        poisoned = replace(target, future={h: 1000.0 for h in range(1, 24)})
        self.assertEqual(expected, model.predict_logs(poisoned, records))
        contaminated = copy.deepcopy(case)
        contaminated["history_hourly"].append({"from": case["forecast"][-1]["from"],
                                                "viewers": 1e9, "channels": 1})
        contaminated["history"].append({"at": case["forecast"][-1]["from"],
                                        "viewers": 1e9, "channels": 1})
        contaminated["answer"] = {"hours": [1e9] * 23}
        self.assertEqual(model.forecast_case(case), model.forecast_case(contaminated))

    def test_training_features_are_censored(self):
        case = copy.deepcopy(CASES[0])
        original = model.launch_records(case)
        for item in case["comparables"]:
            for row in item["hourly"]:
                if model.timestamp(row["from"]) >= model.timestamp(item["day_start"]) + 3600:
                    row["viewers"] = 1e8
                    row["channels"] = 1e6
        changed = model.launch_records(case)
        self.assertEqual([(r.name, r.x, r.meta) for r in original],
                         [(r.name, r.x, r.meta) for r in changed])
        self.assertNotEqual(original[0].future, changed[0].future)

    def test_zero_sparse_and_missing_comparables(self):
        for first in (0, 1, 100000, 1e9):
            case = copy.deepcopy(CASES[0])
            case["comparables"] = []
            case["history_hourly"] = []
            case["history"] = []
            case["known"][0]["viewers"] = first
            case["known"][0]["channels"] = None
            validate(model.forecast_case(case))
        case = copy.deepcopy(CASES[0])
        for item in case["comparables"]:
            item["hourly"] = [r for r in item["hourly"] if int(r["from"][11:13]) != 7]
            for row in item["hourly"]:
                row["channels"] = None
        validate(model.forecast_case(case))

    def test_first_hour_adaptation(self):
        case = copy.deepcopy(CASES[1])
        before = model.forecast_case(case)
        case["known"][0]["viewers"] *= 2
        after = model.forecast_case(case)
        self.assertTrue(all(a["q"][2] > b["q"][2] for a, b in zip(after["hours"], before["hours"])))

    def test_stdlib_only(self):
        for filename in ("main.py", "evaluate.py", "experiments.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module]
                for name in names:
                    self.assertIn(name.split(".")[0], sys.stdlib_module_names | {"main", "evaluate"})

    def test_process_isolation_determinism_and_runtime(self):
        timings, hashes = {}, {}
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as folder:
            base = Path(folder)
            inputs = base / "input"
            outputs = base / "output"
            inputs.mkdir()
            env = os.environ.copy()
            env["MAKEFUN_INPUT_DIR"] = str(inputs)
            env["MAKEFUN_OUTPUT_DIR"] = str(outputs)
            # Each invocation can see only its own case, just like the grader.
            for case in CASES:
                path = inputs / (case["id"] + ".in")
                path.write_text(json.dumps(case), encoding="utf-8")
                start = time.perf_counter()
                subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")],
                               cwd=base, env=env, check=True, timeout=8, capture_output=True)
                timings[case["id"]] = time.perf_counter() - start
                destination = outputs / (case["id"] + ".out")
                self.assertTrue(destination.is_file() and not destination.is_symlink())
                self.assertLessEqual(destination.stat().st_size, 65536)
                content = destination.read_bytes()
                validate(json.loads(content))
                hashes[case["id"]] = content
                path.unlink()
            # Batch operation must agree byte for byte with isolated operation,
            # even under a different hash seed and from a different directory.
            for case in reversed(CASES):
                (inputs / (case["id"] + ".in")).write_text(json.dumps(case), encoding="utf-8")
            env["PYTHONHASHSEED"] = "1234"
            subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")],
                           cwd=base, env=env, check=True, timeout=8, capture_output=True)
            for case in CASES:
                self.assertEqual(hashes[case["id"]], (outputs / (case["id"] + ".out")).read_bytes())
        (ROOT / "runtime-check.json").write_text(json.dumps({"python": sys.version,
            "isolated_process_seconds": timings, "deterministic": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    CASES[:] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.input_dir.glob("*.in"))]
    unittest.main(argv=[sys.argv[0]] + remaining)
