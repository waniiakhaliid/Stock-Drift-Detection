# Watching the Distribution Move - Phase 3 (DAV research project)

Online detection and visualisation of temporal data drift in stock indices (S&P 500 and KSE-100), with a leakage-free protocol,
synthetic ground truth, ablations and an interactive dashboard.
Group: Wania Khalid (24i2608), Eshaal Hussain (24i2509) - DS-B - Submitted to Ma'am Azka Atiq.

## Folder layout

```
phase3_drift/
├── README.md              this file
├── run_all.py             runs the whole pipeline in order (python run_all.py --list)
├── requirements.txt       pip dependencies          pyproject.toml   same, for uv
├── report/
│   └── Phase3_Report_FINAL.docx      the submission report (Sections 1-7, Appendix A)
├── src/                   all code (every script works from any directory)
│   ├── paths.py           the only place folder names are defined
│   ├── drift.py           library: features, KS/MMD, forecaster, ADWIN/PH/BOCPD, baselines, scoring, synthetic generator
│   ├── run_real.py        E2 + E3b on one real index           -> results/results_<NAME>.pkl
│   ├── run_synth.py       E1 + E3a synthetic benchmark (slow)  -> results/synth_results.pkl
│   ├── analyze.py         Tables, Friedman/Nemenyi, separation test, ablations -> results/tables.pkl
│   ├── analyze_kse.py     KSE-100 tables + two-market comparison -> outputs/tables/, Figures 13-15
│   ├── leakage_demo.py    random vs chronological split        -> results/leakage.pkl
│   ├── make_figs.py       Figures 1-12                          -> outputs/figs/
│   ├── dashboard.py       interactive dashboard                 -> outputs/dashboard.html
│   └── user_study.py      E4 stimuli, answer key, response sheet -> user_study/
├── data/                  input CSVs and event calendars (see the Data section below)
├── results/               stored pickles (so figures/tables can be rebuilt without re-running detectors)
├── outputs/               figs/ (15 PNG), tables/ (KSE CSVs), dashboard.html
└── user_study/            E4 kit (see user_study/README.md); no participant data yet
```

## Data

| File | What it is |
|---|---|
| `data/SP500_close.csv` | S&P 500 daily close, 1990-01-02 to 2022-12-28 (8,313 rows), exported from the series bundled with the `skfolio` package. Columns `Date,Close`. |
| `data/KSE100.csv` | KSE-100 daily close, 2004-01-01 to 2022-12-28 (4,697 rows). Columns `Date,Close`. **Source:** PSXpert GitHub repository (`MuHamzaSheikh/PSXpert`, file `data/kse100_index_history_2002_2026.csv`), downloaded on **<DATE>**, reduced to the `date` and `close` columns and filtered to 2004-01-01 - 2022-12-28. Secondary (open-source) source, not an official PSX export; values were not independently verified. No values were changed, filled or interpolated. |
| `data/events_KSE100.csv` | Frozen KSE-100 event calendar (columns `date,label`): five stress-onset dates written before the KSE-100 was run, chosen in hindsight (report Section 2.2). |

The S&P 500 calendar (5 events + 5 post-hoc events) lives inside `src/run_real.py`.
To add another index: put `NAME.csv` (columns `Date,Close`; at least ~4 months before 2005-01-01) and `events_NAME.csv` (columns `date,label`) in `data/`, then
`python src/run_real.py --csv NAME.csv --name NAME --events events_NAME.csv`.

## Setup

Python 3.10-3.12.

```
uv sync                     # or: python -m pip install -r requirements.txt
uv run python run_all.py    # or: python run_all.py
```

With `uv`, always prefix commands with `uv run` (plain `python` uses a different interpreter and will not find the packages).

## Run

| Goal | Command | Time |
|---|---|---|
| Everything except the synthetic run (uses the stored synthetic results) | `python run_all.py` | about 5 min |
| Include the synthetic benchmark (230 series, resumable) | `python run_all.py --force-synth` | long |
| One step | `python run_all.py --steps analyze_kse` | seconds |
| One market | `python src/run_real.py --csv KSE100.csv --name KSE100 --events events_KSE100.csv` | about 30 s |

Fixed seed (42) throughout. Re-running `run_real.py` for both markets and `analyze.py` reproduces the stored results exactly (checked).

## Pipeline in one paragraph

Day *t* uses only data up to *t*. The first two years (2005-2006) are the setup period: feature scaling, the ridge forecaster, the
KS/MMD null distribution (moving-block bootstrap) and the ADWIN / Page-Hinkley / BOCPD thresholds are fitted there and then frozen.
Monitors: 5A KS/MMD on six features (covariate drift), 5B ADWIN + Page-Hinkley on the forecaster's absolute error (concept drift),
5C BOCPD on returns, plus a fusion alarm (>= 2 of 3 families within 10 days). Baselines: online HMM, simplified jump model,
volatility rule, fixed schedule, ADWIN on |returns|, and offline PELT as a hindsight ceiling. Alarms are scored against a frozen
event calendar (precision, recall, false alarms per year, delay/lead) with a random-alarm chance reference, and on synthetic series
with known drift date and type.

## Phase 3 requirements -> where to find them

| # | Requirement | Report | Code / output |
|---|---|---|---|
| 1 | Implementation + changes from Phase 2 | Sec. 1 (change log 1.2) | `src/drift.py`, `run_real.py` |
| 2 | Dataset, preprocessing, splits | Sec. 2 | `data/`, `run_real.py` |
| 3 | Baseline evaluation | Sec. 3 (Tables 1-6) | `analyze.py`, `analyze_kse.py` |
| 4 | Graphs and visualisation | Sec. 4 (Figures 1-15) | `outputs/figs/`, `outputs/dashboard.html` |
| 5 | Comparison with existing research | Sec. 5 | `leakage_demo.py` |
| 6 | Ablations | Sec. 6 (Tables 8-10) | `analyze.py` |
| 7 | Results analysis and discussion | Sec. 7 | - |

## Main results (details in the report)

- Synthetic: KS/MMD and Page-Hinkley give the best chance-adjusted detection (excess 0.25 and 0.24); HMM and jump model look best on raw detection only because they alarm about 3.3 times a year.
- Covariate and concept drift separate on synthetic data (AUC 0.81-1.00 and 0.84-0.92); variance-only changes do not (AUC 0.51-0.61).
- S&P 500: no online detector is precise (precision <= 0.11); only Page-Hinkley, the volatility rule and fusion beat random alarms.
- KSE-100: no online detector beats random alarms (smallest p = 0.108); detector rankings differ from the S&P 500.
- Leakage: a random split of overlapping windows gives F = 0.88; a chronological split with a 20-day purge gives F = 0.28.

## Known limits and open items

- **RQ3 is not answered**: the user-study kit exists but no participants have been run.
- **Study-design issues** (see `user_study/README.md`): the drift is always at day 150 of 300, and the dashboard shows more information than the static plot.
- Close-only data (no volume); five events per market, so recall moves in steps of 0.2.
- KSE-100 data come from a secondary open-source repository (PSXpert on GitHub, see the Data section), not an official PSX export; values were not independently verified.
- Not implemented from Phase 2 (listed in change log rows 12-15): Diebold-Mariano test, forecast RMSE before/after drift, AR(1) sanity forecaster, per-event bootstrap intervals.
- `answer_key_DO_NOT_SHARE.csv` must never be given to participants.

## Troubleshooting

- `ModuleNotFoundError: river` - you used plain `python` with a `uv` environment; use `uv run python ...` or `pip install -r requirements.txt`.
- `hmmlearn`/`ruptures` fail to install - use Python 3.10-3.12.
- A new index fails in `run_real.py` - check the CSV has columns `Date,Close`, ascending dates, and data from before 2005.