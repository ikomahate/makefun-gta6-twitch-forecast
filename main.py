"""Conditional launch-curve forecasts, Python 3.13 standard library only.

The input's comparable launches supply training labels. The current game's
history is cut at its checkpoint before features are made. Everything is
modelled in natural logarithms, the units of the contest's pinball loss.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
LOG_MAX = math.log(1_000_000_000)
RIDGE = 0.3
CONSOLE_RESIDUAL_SCALE = 0.5


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def log_count(value):
    return math.log(max(1.0, float(value or 0)))


def clip(value, low, high):
    return max(low, min(high, value))


def metadata(platforms, unlock):
    """Use platform/unlock facts, never research notes containing future peaks."""
    p = platforms.lower()
    u = unlock.lower().split("source:")[0].split("tier:")[0]
    console = ("pc" not in p and "stadia" not in p) or "console only" in p
    early = any(word in u for word in ("early access", "early-access", "premium", "deluxe"))
    if "no early access" in u or "no early-access" in u:
        early = False
    simultaneous = "single global" in u or "single simultaneous" in u or "simultaneous across" in u
    return (float(console), float(early), float(simultaneous))


def case_metadata(case):
    notes = case.get("notes", "")
    platforms = notes.split("real unlock:")[0]
    if "Grand Theft Auto VI" == case.get("game", {}).get("name"):
        platforms = "PS5, Xbox Series X|S"
        notes = "Rolling midnight local; no early access."
    return metadata(platforms, notes.split("real unlock:")[-1])


def hourly_rows(rows, start, cutoff=None):
    """Map actual sampled hours; never interpolate across missing hours."""
    buckets = {}
    for row in rows or []:
        at = timestamp(row.get("from", row.get("at")))
        if cutoff is not None and at >= cutoff:
            continue
        h = math.floor((at - start) / 3600)
        buckets.setdefault(h, []).append(row)
    result = {}
    for h, group in buckets.items():
        viewers = [r["viewers"] for r in group if r.get("viewers") is not None]
        channels = [r["channels"] for r in group if r.get("channels") is not None]
        if viewers:
            result[h] = (mean(viewers), mean(channels) if channels else None)
    return result


def features(history, first):
    base = log_count(first)

    def average_log(hours, field=0):
        values = [log_count(history[h][field]) for h in hours
                  if h in history and history[h][field] is not None]
        return mean(values) if values else (base if field == 0 else 0.0)

    last = average_log(range(-2, 0)) - base
    evening = average_log(range(-8, -2)) - base
    recent = [log_count(history[h][0]) for h in range(-12, 0) if h in history]
    peak = max(recent, default=base) - base
    channel0 = history.get(0, (first, None))[1]
    channel_growth = 0.0
    if channel0 is not None and any(h in history and history[h][1] is not None for h in range(-6, -1)):
        channel_growth = log_count(channel0) - average_log(range(-6, -1), 1)
    older = average_log(range(-72, -24)) - base
    return tuple(clip(x, -4, 4) for x in
                 (peak, evening, last, channel_growth, (base - 11.5) / 2, older))


@dataclass
class Launch:
    name: str
    history: dict
    first: float
    x: tuple
    meta: tuple
    future: dict


def launch_records(case):
    records = []
    for item in case.get("comparables", []):
        if "target-history" in item.get("unlock", "").lower():
            continue
        if item["name"].casefold() == case.get("game", {}).get("name", "").casefold():
            continue
        start = timestamp(item["day_start"])
        history = hourly_rows(item.get("hourly"), start, start + 3600)
        if 0 not in history or history[0][0] <= 0:
            continue
        full = hourly_rows(item.get("hourly"), start)
        future = {h: log_count(full[h][0]) - log_count(history[0][0])
                  for h in range(1, 24) if h in full}
        if len(future) < 18:
            continue
        first = history[0][0]
        records.append(Launch(item["name"], history, first, features(history, first),
                              metadata(item.get("platforms", ""), item.get("unlock", "")), future))
    return records


def target_record(case):
    start = timestamp(case["day_start"])
    cutoff = min(timestamp(case["checkpoint"]), start + 3600)
    history = hourly_rows(case.get("history_hourly") or case.get("history"), start, cutoff)
    known = next(row for row in case["known"] if row["hour"] == 1)
    first = max(1.0, float(known["viewers"]))
    history[0] = (first, known.get("channels"))
    return Launch(case.get("game", {}).get("name", ""), history, first,
                  features(history, first), case_metadata(case), {})


def analogue_weights(target, records, bandwidth=1.0):
    result = []
    for record in records:
        if record.name == target.name:
            continue
        distance = sum(s * (a - b) ** 2 for s, a, b in
                       zip((1.3, 0.7, 0.3, 0.2, 0.1, 0.05), target.x, record.x))
        distance += 2.5 * (target.meta[0] - record.meta[0]) ** 2
        distance += 2.5 * (target.meta[1] - record.meta[1]) ** 2
        distance += 5.0 * (target.meta[2] - record.meta[2]) ** 2
        distance += 8.0 * (float(target.first >= 5000) - float(record.first >= 5000)) ** 2
        result.append((record, math.exp(-min(700, distance / (2 * bandwidth ** 2)))))
    total = sum(w for _, w in result)
    return [(r, w / total) for r, w in result] if total else []


def mixture_quantile(points, spread, probability):
    """Invert a smoothed empirical CDF in log space; monotone by construction."""
    total = sum(w for _, w in points)
    low = min(x for x, _ in points) - 9 * spread
    high = max(x for x, _ in points) + 9 * spread
    scale = math.sqrt(2) * spread
    for _ in range(42):
        middle = (low + high) / 2
        cdf = sum(w * (1 + math.erf((middle - x) / scale)) * 0.5 for x, w in points) / total
        if cdf < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def inverse(matrix):
    """Small positive-definite ridge system, solved with partial pivoting."""
    n = len(matrix)
    rows = [list(row) + [float(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        divisor = rows[i][i]
        rows[i] = [x / divisor for x in rows[i]]
        for j in range(n):
            if j != i:
                multiple = rows[j][i]
                rows[j] = [a - multiple * b for a, b in zip(rows[j], rows[i])]
    return [row[n:] for row in rows]


def local_adjustments(target, analogues, ridge, residual_scale=1.0):
    """Learn a conservative linear transport between comparable trajectories.

    Weights sum to one: ridge is in covariance units rather than depending on
    the number of comparables. The same factorization serves all 23 horizons.
    """
    size = len(target.x)
    center = [sum(w * r.x[j] for r, w in analogues) for j in range(size)]
    centered = [[x - c for x, c in zip(r.x, center)] for r, _ in analogues]
    covariance = [[sum(w * v[i] * v[j] for v, (_, w) in zip(centered, analogues))
                   + (ridge if i == j else 0) for j in range(size)] for i in range(size)]
    inv = inverse(covariance)
    shift = [a - b for a, b in zip(target.x, center)]
    result = {}
    for h in range(1, 24):
        valid = [(r, w, v) for (r, w), v in zip(analogues, centered) if h in r.future]
        total = sum(w for _, w, _ in valid)
        if not total:
            continue
        ymean = sum(w * r.future[h] for r, w, _ in valid) / total
        cross = [sum(w * v[j] * (r.future[h] - ymean) for r, w, v in valid)
                 for j in range(size)]
        beta = [sum(a * b for a, b in zip(row, cross)) for row in inv]
        prediction = ymean + sum(b * x for b, x in zip(beta, shift))
        result[h] = []
        for r, w, v in valid:
            fitted = ymean + sum(b * x for b, x in zip(beta, v))
            # Do not use in-sample residuals at face value. The ridge hat
            # diagonal supplies an analytic leave-one-out correction.
            leverage = w * (1 + sum(v[i] * sum(inv[i][j] * v[j] for j in range(size))
                                    for i in range(size)))
            residual = (r.future[h] - fitted) / max(0.35, 1 - leverage)
            result[h].append((prediction + residual_scale * residual, w))
    return result


def predict_logs(target, records, bandwidth=1.0, ridge=RIDGE, adjustment=1.0,
                 spread_scale=1.0, residual_scale=None):
    analogues = analogue_weights(target, records, bandwidth)
    if residual_scale is None:
        residual_scale = CONSOLE_RESIDUAL_SCALE if target.meta[0] and not target.meta[1] else 1.0
    adjusted = local_adjustments(target, analogues, ridge, residual_scale) if analogues and adjustment else {}
    # GTA can exceed every historical launch. Do not let calibration on smaller
    # launches imply unjustified precision outside the observed scale range.
    largest = max((r.first for r, _ in analogues), default=target.first)
    extrapolation = 0.15 * max(0.0, math.log(target.first / max(1.0, largest)))
    result = {}
    for h in range(1, 24):
        points = [(r.future[h], w) for r, w in analogues if h in r.future]
        if h in adjusted:
            points = [((1 - adjustment) * original[0] + adjustment * transported[0], original[1])
                      for original, transported in zip(points, adjusted[h])]
        if not points:
            center = -0.45 * math.sin(math.pi * min(h, 18) / 18)
            points = [(center, 1.0)]
            spread = 0.18 + 0.12 * math.sqrt(h)
        else:
            spread = math.hypot(spread_scale * (0.08 + 0.015 * math.sqrt(h)), extrapolation)
        result[h] = [log_count(target.first) + mixture_quantile(points, spread, q) for q in LEVELS]
    return result


def forecast_case(case):
    logs = predict_logs(target_record(case), launch_records(case))
    return {"hours": [{"hour": h + 1, "q": [round(math.exp(clip(v, 0, LOG_MAX)), 6) for v in logs[h]]}
                      for h in range(1, 24)]}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        output = forecast_case(case)
        destination = output_dir / (path.stem + ".out")
        encoded = json.dumps(output, allow_nan=False, separators=(",", ":")) + "\n"
        # Atomic replacement guarantees a regular output even if an old output
        # path was a link; it also prevents leaving a partially written JSON.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                         prefix=".forecast-", delete=False) as temporary:
            temporary.write(encoded)
            temporary_path = temporary.name
        os.replace(temporary_path, destination)


if __name__ == "__main__":
    main()
