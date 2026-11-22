"""leakage_demo.py - comparison with An et al. (2025): random split of overlapping windows vs a chronological,
purged split, on the same S&P 500 data. Task: warn 20 days ahead of a volatility-regime switch."""
import pickle, numpy as np, pandas as pd
import paths as P
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import drift as D
R = pickle.load(open(P.RESULTS/"results_SP500.pkl", "rb"))
F, n, ns = R["F"], R["n"], R["n_setup"]
vol = F["vol21"].values
thr = 2 * np.median(vol[:ns])
H = 20
# label: 21-day volatility is above 2x its setup median at some day in t+1..t+20 while it is below it at t
lab = np.zeros(n, int)
for t in range(n - H):
    lab[t] = int(vol[t] <= thr and (vol[t + 1:t + H + 1] > thr).any())
# features: the 6 z-scored features at t, t-20, t-60  (a 'window' view like An et al.)
Z = R["Z"]
X = np.hstack([Z[60:-H], Z[40:-H - 20], Z[0:-H - 60]]) if False else np.hstack([Z[60:n - H], Z[40:n - H - 20], Z[0:n - H - 60]])
y = lab[60:n - H]
m = (vol[60:n - H] <= thr)                 # only days where the regime is currently 'calm' can switch up
X, y = X[m], y[m]
idx = np.flatnonzero(m) + 60
print("samples", len(y), "positives", y.sum())
res = {}
rng = np.random.default_rng(0)
for name in ("random split (An et al. style)", "chronological + 20-day purge"):
    sc = []
    for rep in range(10):
        if name.startswith("random"):
            p = rng.permutation(len(y)); k = int(0.8 * len(y)); tr, te = p[:k], p[k:]
        else:
            k = int(0.8 * len(y)); tr = np.arange(k)
            te = np.arange(k + H, len(y))
        clf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=rep, n_jobs=1)
        clf.fit(X[tr], y[tr]); pr = clf.predict(X[te])
        sc.append([f1_score(y[te], pr, zero_division=0), accuracy_score(y[te], pr),
                   precision_score(y[te], pr, zero_division=0), recall_score(y[te], pr, zero_division=0)])
        if not name.startswith("random"):
            break
    sc = np.array(sc)
    res[name] = dict(f1=sc[:, 0].mean(), acc=sc[:, 1].mean(), prec=sc[:, 2].mean(), rec=sc[:, 3].mean(),
                     base_rate=float(y[te].mean()))
    print(name, {k: round(v, 3) for k, v in res[name].items()})
# naive baseline: always predict 'switch'
te_b = y[int(0.8 * len(y)) + H:]
res["always-warn baseline (chronological test)"] = dict(f1=f1_score(te_b, np.ones_like(te_b)), acc=accuracy_score(te_b, np.ones_like(te_b)),
                                                         prec=te_b.mean(), rec=1.0, base_rate=float(te_b.mean()))
pickle.dump(res, open(P.RESULTS/"leakage.pkl", "wb"))
