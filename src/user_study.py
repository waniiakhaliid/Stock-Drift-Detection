"""user_study.py - E4 kit: stimuli, Latin-square schedule, response template.
analyze_study.py (below, written by this script) analyses the collected responses.
NO participant data exist yet: results for RQ3 must be collected."""
import pickle, numpy as np, pandas as pd
import paths as P
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import dashboard as DB
R = pickle.load(open(P.RESULTS/"results_SP500.pkl", "rb"))
ev, close, F = R["events"], R["close"].values, R["F"]
L, POS = 300, 150
sets = {"A": ["2008 Lehman collapse", "2022 rate-hike selloff"], "B": ["2020 COVID crash onset", "2011 US downgrade selloff"]}
key, wid = [], 0
for s, evs in sets.items():
    for e in evs:
        wid += 1
        lo = ev[e] - POS; hi = lo + L
        w = f"W{wid}"
        key.append(dict(window=w, set=s, true_drift_day=POS, event=e, start=str(R["dates"][lo].date())))
        fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
        ax[0].plot(close[lo:hi] / close[lo] * 100, color="#1f4e79"); ax[0].set_ylabel("price (rebased)")
        ax[1].plot(F["vol21"].values[lo:hi], color="#e08e0b"); ax[1].set_ylabel("21-d volatility"); ax[1].set_xlabel("trading day in window")
        fig.suptitle(f"Window {w} (static plot)"); fig.tight_layout(); fig.savefig(P.STUDY/f"stimuli/{w}_static.png", dpi=130); plt.close()
        open(P.STUDY/f"stimuli/{w}_dashboard.html", "w", encoding="utf-8").write(DB.build(lo, hi, True, True, f"Window {w} (dashboard)"))
pd.DataFrame(key).to_csv(P.STUDY/"answer_key_DO_NOT_SHARE.csv", index=False)
# Latin square: 4 sequences, participants cycle through them
seqs = {1: [("A", "static"), ("B", "dashboard")], 2: [("B", "static"), ("A", "dashboard")],
        3: [("A", "dashboard"), ("B", "static")], 4: [("B", "dashboard"), ("A", "static")]}
rows = []
for p in range(1, 17):
    sq = (p - 1) % 4 + 1
    for order, (s, cond) in enumerate(seqs[sq], 1):
        for w in [k["window"] for k in key if k["set"] == s]:
            rows.append(dict(participant=p, sequence=sq, order=order, condition=cond, set=s, window=w,
                             marked_day="", seconds="", confidence_1to5=""))
pd.DataFrame(rows).to_csv(P.STUDY/"response_sheet_TEMPLATE.csv", index=False)
open(P.STUDY/"analyze_study.py", "w").write('''"""Analyse collected responses. Usage: python analyze_study.py response_sheet_filled.csv
Hit = marked day within +-10 trading days of the true drift day (day 150)."""
import sys, numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(sys.argv[1]); d["abs_err"] = (d.marked_day - 150).abs(); d["hit"] = (d.abs_err <= 10).astype(float)
g = d.groupby(["participant", "condition"]).agg(hit=("hit", "mean"), err=("abs_err", "mean"), sec=("seconds", "mean"),
                                               conf=("confidence_1to5", "mean")).reset_index()
rng = np.random.default_rng(0)
for m in ("hit", "err", "sec", "conf"):
    w = g.pivot(index="participant", columns="condition", values=m).dropna()
    diff = (w["dashboard"] - w["static"]).values
    res = stats.wilcoxon(diff)
    rb = (diff > 0).mean() - (diff < 0).mean()                      # matched-pairs rank-biserial (simple form)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    print(f"{m}: static={w['static'].mean():.2f} dashboard={w['dashboard'].mean():.2f} W={res.statistic:.1f} p={res.pvalue:.4f} "
          f"effect(r_b)={rb:.2f} mean diff={diff.mean():.2f} 95% CI [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] n={len(diff)}")
''')
print(len(rows), "rows")
