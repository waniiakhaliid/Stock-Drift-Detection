"""analyze_kse.py - tables and figures for a second market (KSE-100) + cross-market comparison.
Usage: python analyze_kse.py [KSE100]   (needs results_KSE100.pkl and results_SP500.pkl in this folder)"""
import sys, os, pickle
import paths as P
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NAME = sys.argv[1] if len(sys.argv) > 1 else "KSE100"
R = pickle.load(open(P.RESULTS/f"results_{NAME}.pkl", "rb")); S = pickle.load(open(P.RESULTS/"results_SP500.pkl", "rb"))
P.FIGS.mkdir(parents=True, exist_ok=True); P.TABLES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
C = {"blue": "#1f4e79", "red": "#c0392b", "grey": "#7f8c8d", "purple": "#6c3483", "orange": "#e08e0b"}
names = ["KS/MMD (features)", "ADWIN (forecast error)", "Page-Hinkley (forecast error)", "BOCPD (returns)",
         "Fusion (>=2 of 3 families)", "HMM (online, 2-state)", "Jump model (simplified)", "Volatility rule",
         "Fixed schedule (252 d)", "ADWIN on |returns| (Neri-style)", "PELT offline (hindsight ceiling)"]
f = lambda v: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)
# Table K1: main results at +-10 d with chance reference
rows = []
for k in names:
    s, c = R["score"][10][k], R["chance"][k]
    rows.append([k, int(s["n_alarms"]), f(s["precision"]), f(s["recall"]), f(s["far"]), f(s["median_delay"]), f(c["exp_recall"]), round(c["p_recall"], 3)])
K1 = pd.DataFrame(rows, columns=["Detector", "Alarms", "Precision", "Recall", "FAR/yr", "Median delay (d)", "Chance recall", "p"])
# Table K2: tolerance sensitivity (recall / precision)
K2 = pd.DataFrame({k: {f"+-{t} d": f"{R['score'][t][k]['recall']:.1f} / {R['score'][t][k]['precision']:.2f}" for t in (5, 10, 20)} for k in names}).T
# Table K3: cross-market (+-10 d): recall / precision / FAR
K3 = pd.DataFrame({k: {"S&P 500 recall": f(S["score"][10][k]["recall"]), "S&P 500 prec.": f(S["score"][10][k]["precision"]), "S&P 500 FAR": f(S["score"][10][k]["far"]),
                       f"{NAME} recall": f(R["score"][10][k]["recall"]), f"{NAME} prec.": f(R["score"][10][k]["precision"]), f"{NAME} FAR": f(R["score"][10][k]["far"])} for k in names}).T
for nm, t in (("K1", K1), ("K2", K2), ("K3", K3)):
    t.to_csv(P.TABLES/f"{NAME}_{nm}.csv", index=nm != "K1"); print(f"\n== {NAME} {nm} ==\n", t.to_string())
dates, n, ns, ev = R["dates"], R["n"], R["n_setup"], R["events"]
print("\nparams", {k: v for k, v in R["params"].items() if k != "ridge_w"}, "| n", n, "setup", ns)
print("events idx->date", {k: str(dates[i].date()) for k, i in ev.items()})
print("\nE3b gaps (alarm day minus error-rise day):\n", R["gaps"].to_string())
print("granger (ks->err p, err->ks p):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in R["granger"].items()})
lags, ccf = R["ccf"]; print("CCF peak lag", lags[int(np.nanargmax(ccf))], "max", round(float(np.nanmax(ccf)), 3))
print("Mirza-style tests:", {k: {m: round(float(x), 4) for m, x in v.items()} for k, v in R["mirza"]["tests"].items()}, {k: round(float(v), 3) for k, v in R["mirza"]["mae"].items()})
# Fig 13: overview
fig, ax = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
ax[0].plot(dates, R["close"].values, color=C["blue"], lw=0.9); ax[0].set_ylabel(f"{NAME} close")
ax[1].plot(dates, R["F"]["vol21"].values, color=C["orange"], lw=0.9); ax[1].set_ylabel("21-d volatility")
ax[2].plot(dates, R["ks_score"], color=C["purple"], lw=0.9, label="max KS stat"); ax[2].plot(dates, R["err_roll"], color=C["red"], lw=0.9, label="21-d forecast error"); ax[2].legend(fontsize=7)
for a in ax:
    for k, i in ev.items(): a.axvline(dates[i], color=C["red"], ls="--", lw=0.8)
    a.axvspan(dates[0], dates[ns], color=C["grey"], alpha=0.15)
fig.suptitle(f"Figure 13. {NAME}: price, volatility and drift signals with the frozen event calendar (dashed)", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/f"fig13_{NAME}_overview.png"); plt.close()
# Fig 14: alarm raster
fig, ax = plt.subplots(figsize=(9, 5))
for j, nm in enumerate(names):
    al = R["alarms"][nm]; col = C["blue"] if j < 5 else (C["grey"] if j < 10 else C["purple"])
    ax.scatter([dates[a] for a in al], [j] * len(al), marker="|", s=110, color=col, lw=1.6)
for k, i in ev.items(): ax.axvspan(dates[max(i - 10, 0)], dates[min(i + 10, n - 1)], color=C["red"], alpha=0.25)
ax.axvspan(dates[0], dates[ns], color=C["grey"], alpha=0.15)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
ax.set_title(f"Figure 14. {NAME}: alarm dates of every detector\nRed bands: calendar events +-10 trading days; grey: setup period", fontsize=9)
fig.tight_layout(); fig.savefig(P.FIGS/f"fig14_{NAME}_alarms.png"); plt.close()
# Fig 15: metrics vs chance, both markets side by side
fig, ax = plt.subplots(2, 3, figsize=(10, 7.5), sharey=True)
y = np.arange(len(names)); cols = [C["blue"] if j < 5 else (C["grey"] if j < 10 else C["purple"]) for j in range(len(names))]
for r_, (M, lab) in enumerate(((S, "S&P 500"), (R, NAME))):
    sc, ch = M["score"][10], M["chance"]
    ax[r_, 0].barh(y, [sc[k]["recall"] for k in names], color=cols); ax[r_, 0].scatter([ch[k]["exp_recall"] for k in names], y, color="k", marker="D", s=14, zorder=3)
    ax[r_, 1].barh(y, [sc[k]["precision"] for k in names], color=cols); ax[r_, 1].scatter([ch[k]["exp_precision"] for k in names], y, color="k", marker="D", s=14, zorder=3)
    ax[r_, 2].barh(y, [sc[k]["far"] for k in names], color=cols)
    ax[r_, 0].set_ylabel(lab, fontweight="bold"); ax[r_, 0].set_yticks(y); ax[r_, 0].set_yticklabels(names, fontsize=7)
for c_, t in enumerate(("Recall (+-10 d; diamonds = random alarms)", "Precision", "False alarms per year")): ax[0, c_].set_title(t, fontsize=8)
ax[0, 0].invert_yaxis(); fig.suptitle(f"Figure 15. Detection quality in two markets: S&P 500 (top) and {NAME} (bottom)", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/f"fig15_two_markets.png"); plt.close()
print("\nsaved tables/ and figs/")
