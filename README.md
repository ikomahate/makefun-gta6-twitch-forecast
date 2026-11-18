# GTA VI Twitch launch forecasting

An offline entry for `gta6-twitch-ccu-forecast-01`. `main.py` uses the Python
standard library, reads each case independently, and writes five ordered
quantiles for every hour from 2 through 24. No packages, network calls, saved
practice answers, credentials, or wallet access are needed.

The starter has been replaced with a conditional statistical model. All
training data comes from the current input's historical comparables. No
submission has been made.

## Run

The grader runs `python3 -P -s /bot/main.py` with Python 3.13. Its defaults are
`/input` and `/output`. For a local PowerShell run from this repository:

```powershell
$env:MAKEFUN_INPUT_DIR = '..\gta6-twitch-ccu-forecast-01-kit\input'
$env:MAKEFUN_OUTPUT_DIR = '.\out'
python -P -s main.py
```

The files `out/p00.out` through `out/p03.out` contain the practice forecasts.
For each input `<id>.in`, output is `<id>.out`, with exactly the competition's
`hours`, `hour`, and `q` fields. Files are written atomically.

## Method

Each comparable's release-day trajectory is expressed as log viewer ratios to
its own first hour. Features use only data before its checkpoint: recent peak,
evening activity, the last two hours, channel growth, launch scale, and older
activity. The current case's official first-hour print supplies the anchor.
Minute histories are aggregated by UTC hour; absent hours stay absent.

Comparable weights combine trajectory similarity with console versus PC,
early-access, and simultaneous versus rolling-unlock metadata. A local ridge
regression adjusts each comparable curve to the current game's observed
features. The remaining residual curves form a predictive mixture. An analytic
leave-one-out leverage correction addresses regression residual shrinkage;
an additional residual scale is chosen using held-out log pinball loss.
Gaussian smoothing supplies continuous quantiles and an uncertainty floor.
Uncertainty increases if the first hour exceeds every training launch's scale.

The five quantiles are obtained by numerically inverting the mixture CDF in log
space. This supports asymmetric uncertainty and guarantees ordered quantiles.
Output is finite and bounded from 1 to 1 billion viewers. Missing comparables
trigger a broad, first-hour-anchored fallback with an overnight dip and rebound.

The model uses no case-specific forecast tables. In particular, `main.py` never
reads evaluation labels, reports, or the other practice cases.

## Historical findings

The supplied SullyGnome data contains 37 launch comparables and one separate GTA
VI pre-launch history. Seven comparable launches are console-only without early
access. Across those seven, the median ratio to hour 1 is about 0.82 in hour 2,
0.52 in hour 4, 1.00 in hour 6, and 2.66 in hour 19. These cohort medians describe
the data; the model conditions each curve on its own history and launch scale.

Rolling unlocks and regional viewing schedules overlap. The rules place GTA
VI's New Zealand unlock at 11:00 UTC on November 18, central Europe at 23:00,
the UK at 00:00, and US east/west at 05:00/08:00 on November 19. European
overnight attrition, the American unlocks, and the later European/American
daytime audience make a flat UTC trajectory implausible. Analogues retain
their actual UTC curves, including their release dates' seasonal clock effects.
Different regional rules, PC unlocks, early access, and streamer schedules
remain sources of uncertainty; the model does not assume exact phase alignment.

GTA VI's supplied 632-point pre-launch history peaks at 1,418,094 viewers during
the August 27 premiere. Its sampled median is 24.5, while the November 16 onward
sample median is 1,860 and its maximum is 5,785. Sampling is sparse, so these
medians are not time-weighted averages. The premiere demonstrates event interest
but supplies no defensible fixed launch-day multiplier. This event series is
therefore excluded from launch training. GTA's new first hour and recent launch
history determine the scale at inference time.

Counts and unlock annotations are from the provided `input/*.in`, `NOTICE.txt`,
and `RULES.txt`. No outside information was fetched. `analysis.json` records the
cohort ratios, GTA history summary, and practice analogue weights.

## Measured improvement

The kit explicitly withholds the official practice future counts on the server.
It provides no local practice scorer or answers. Consequently, **there are no
measured official p00–p03 losses in this repository**. All four forecasts have
been generated and validated, but scoring them requires those answers.

Development instead used leave-one-launch-out backtests on the supplied
comparables: the held-out game's entire record is removed from training.
Reported loss is the rules' mean of `2 * max(tau*d, (tau-1)*d)`, where
`d = ln(actual) - ln(prediction)`.

| Version or evaluation | Console log pinball loss | 90% coverage |
| --- | ---: | ---: |
| House persistence baseline | 0.343470 | 77.64% |
| V1: conditional analogue mixture | 0.193100 | 98.76% |
| V2: local regression adjustment | 0.169751 | 98.76% |
| Final: calibrated residual mixture | **0.150112** | 85.71% |
| Nested parameter-selection check | 0.154517 | 84.47% |

The final model reduces loss by 56.3% on these seven console launches, or 55.0%
under the nested selection check. Its typical median miss is 25.86%, compared
with persistence's 72.99%. It beats persistence on each of the seven launches.
Seven launches is a small sample; this is evidence of model development, not a
claim about official practice ranking or unseen GTA performance.

Across all 37 comparables, including PC/global and early-access launches, final
loss is 0.436727 versus persistence's 0.929650. The console cohort is the more
relevant comparison for this task. These baseline numbers differ from the
rules' quoted approximately 0.387, which describes the four official practice
cases rather than our comparable backtests.

The nested check removes the outer test launch from both model training and
inner parameter selection. Its parameters are selected using the other six
console launches. This reduces selection bias within that grid; it is not an
untouched external validation set. Model-family development also saw these
comparables, and the grids were explored sequentially.

The measured revisions are preserved as local artifacts in `evaluation-v1.json`,
`evaluation-v2.json`, `evaluation.json`, `experiments.json` (initial bandwidth
grid), and `experiments-calibration.json` (residual calibration and nested check).
Generated reports and forecasts are excluded from the source snapshot by
`.gitignore`; the scripts recreate current results. Earlier reports reflect their then-current implementations and analytic
baseline; the final evaluator reproduces the house's cent rounding too.

## Reproduce evaluation

```powershell
$caseDir = '..\gta6-twitch-ccu-forecast-01-kit\input'
python -P -s analyze.py --input-dir $caseDir
python -P -s evaluate.py --input-dir $caseDir
python -P -s experiments.py --input-dir $caseDir --stage calibration --nested
python -P -s test_forecast.py --input-dir $caseDir -v
```

`experiments.py --stage weights --nested` reruns the initial bandwidth grid.
Experimental configuration flags affect backtests; production forecasts always
use the constants in `main.py`.

If official practice answers become available locally, place `<id>.json` files
in a separate directory, using this shape with hours 2 through 24:

```json
{"hours":[{"hour":2,"viewers":123456}]}
```

Then run `evaluate.py --input-dir <inputs> --answers-dir <answers>`. Labels are
read only after predictions have been generated and validated. Missing label
files are explicitly marked unscored; missing/null hourly prints are skipped.
No official grading or submission is attempted by these scripts.

## Execution checks and limitations

Seven tests pass, covering the scoring worked example, future-label isolation,
training feature censorship, sparse/zero inputs, first-hour adaptation, standard
library imports, and isolated-process execution. The isolated and batch runs
match byte for byte even under a different hash seed. Every practice output has
23 valid rows and is below the size cap.

Local isolated runs took 0.57–0.63 seconds per case, including interpreter start,
well below eight seconds on this machine. `runtime-check.json` records them.
Only Python **3.12.10** is installed here: the code uses standard-library APIs
available in Python 3.13, but a run in the grader's actual 3.13 Linux container
has not been performed. Sandbox execution times may differ.

Before any later submission, retain the original harness-recorded session logs
and follow the rules' transcript/run-record and repository checks. No logs have
been fabricated, and no account, wallet, network, or submission operations were
performed during this build.
