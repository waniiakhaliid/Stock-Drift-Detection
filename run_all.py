"""Run the whole Phase 3 pipeline in the right order.

    python run_all.py                  # everything except the slow synthetic run if its results already exist
    python run_all.py --force-synth    # also (re)run the synthetic benchmark (slow, resumable)
    python run_all.py --steps analyze_kse make_figs
    python run_all.py --list
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STEPS = {   # name: (script, args, what it does)
    "real_sp500":  ("run_real.py", [], "E2/E3b on the S&P 500 (data/SP500_close.csv)"),
    "real_kse100": ("run_real.py", ["--csv", "KSE100.csv", "--name", "KSE100", "--events", "events_KSE100.csv"], "E2/E3b on the KSE-100"),
    "synth":       ("run_synth.py", ["25"], "E1/E3a synthetic benchmark (slow, resumable)"),
    "analyze":     ("analyze.py", [], "Tables 1-3, 7-10: Friedman/Nemenyi, separation test, ablations"),
    "analyze_kse": ("analyze_kse.py", ["KSE100"], "Tables 4-6 and Figures 13-15 (KSE-100, two-market comparison)"),
    "leakage":     ("leakage_demo.py", [], "Random vs chronological split experiment"),
    "make_figs":   ("make_figs.py", [], "Figures 1-12"),
    "dashboard":   ("dashboard.py", [], "Interactive dashboard -> outputs/dashboard.html"),
    "user_study":  ("user_study.py", [], "Regenerate E4 stimuli, answer key and response sheet"),
}
ap = argparse.ArgumentParser()
ap.add_argument("--steps", nargs="*", choices=STEPS, help="run only these steps (in pipeline order)")
ap.add_argument("--force-synth", action="store_true", help="run the synthetic benchmark even if results/synth_results.pkl exists")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()
if a.list:
    for k, (s, g, w) in STEPS.items(): print(f"{k:12s} {w}")
    sys.exit(0)
todo = [k for k in STEPS if (not a.steps or k in a.steps)]
if not a.steps and not a.force_synth and (ROOT / "results" / "synth_results.pkl").exists():
    todo.remove("synth")
for k in todo:
    script, args, what = STEPS[k]; t = time.time(); print(f"\n=== {k}: {what}", flush=True)
    r = subprocess.run([sys.executable, str(SRC / script), *args], cwd=ROOT)
    if r.returncode: sys.exit(f"step '{k}' failed (exit {r.returncode}); fix it and re-run with --steps {k}")
    print(f"--- {k} done in {time.time() - t:.0f} s", flush=True)
