"""make_figs.py - all figures for the report (figs/*.png)"""
import os, pickle
import paths as P
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import drift as D

P.FIGS.mkdir(parents=True, exist_ok=True)
R = pickle.load(open(P.RESULTS/"results_SP500.pkl", "rb"))
S = pickle.load(open(P.RESULTS/"synth_results.pkl", "rb"))
O = pickle.load(open(P.RESULTS/"tables.pkl", "rb"))
LK = pickle.load(open(P.RESULTS/"leakage.pkl", "rb"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "figure.dpi": 100, "savefig.dpi": 200})
C = {"blue": "#1f4e79", "red": "#c0392b", "green": "#2e8b57", "orange": "#e08e0b", "grey": "#7f8c8d", "purple": "#6c3483"}
dates = R["dates"]; n = R["n"]; ns = R["n_setup"]
ev = R["events"]; ex = R["extra"]
short = {"2008 Lehman collapse": "2008 Lehman", "2011 US downgrade selloff": "2011 downgrade",
         "Dec 2018 selloff": "Dec 2018", "2020 COVID crash onset": "2020 COVID", "2022 rate-hike selloff": "2022 rates"}


def evlines(ax, lab=False):
    for k, i in ev.items():
        ax.axvline(dates[i], color=C["red"], lw=0.9, ls="--", alpha=0.8)
        if lab:
            ax.text(dates[i], ax.get_ylim()[1], short[k], rotation=90, va="top", ha="right", fontsize=7, color=C["red"])


# ---------------- Fig 1: data and features ----------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1, 1]})
ax[0].plot(dates, R["close"].values, color=C["blue"], lw=0.9); ax[0].set_yscale("log")
ax[0].set_ylabel("S&P 500 (log)"); ax[0].axvspan(dates[0], dates[ns], color=C["grey"], alpha=0.2)
ax[0].text(dates[5], R["close"].max() * 0.6, "setup period\n(2005-06)", fontsize=7)
ax[1].plot(dates, R["F"]["vol21"].values, color=C["orange"], lw=0.8); ax[1].set_ylabel("21-day volatility (%)")
ax[2].plot(dates, R["ks_score"], color=C["purple"], lw=0.8); ax[2].set_ylabel("max KS statistic\n(63 d vs 252 d)")
ax[3].plot(dates, R["err_roll"], color=C["green"], lw=0.8); ax[3].set_ylabel("21-day mean\nforecast error")
for a in ax:
    evlines(a)
for k, i in ev.items():
    ax[0].text(dates[i], R["close"].max() * 1.05, short[k], rotation=60, fontsize=7, color=C["red"], ha="left")
ax[0].set_ylim(top=R["close"].max() * 1.7)
fig.suptitle("Figure 1. S&P 500, drift signals and the frozen event calendar (dashed)", y=0.995, fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig1_overview.png"); plt.close()

# ---------------- Fig 2: alarm raster --------------------------------------------------------------
names = ["KS/MMD (features)", "ADWIN (forecast error)", "Page-Hinkley (forecast error)", "BOCPD (returns)",
         "Fusion (>=2 of 3 families)", "HMM (online, 2-state)", "Jump model (simplified)", "Volatility rule",
         "Fixed schedule (252 d)", "ADWIN on |returns| (Neri-style)", "PELT offline (hindsight ceiling)"]
fig, ax = plt.subplots(figsize=(9, 5))
for j, nm in enumerate(names):
    al = R["alarms"][nm]
    col = C["blue"] if j < 5 else (C["grey"] if j < 10 else C["purple"])
    ax.scatter([dates[a] for a in al], [j] * len(al), marker="|", s=110, color=col, lw=1.6)
for k, i in ev.items():
    ax.axvspan(dates[max(i - 10, 0)], dates[min(i + 10, n - 1)], color=C["red"], alpha=0.25)
for k, i in ex.items():
    ax.axvline(dates[i], color=C["orange"], lw=0.7, ls=":")
ax.axvspan(dates[0], dates[ns], color=C["grey"], alpha=0.15)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
ax.set_title("Figure 2. Alarm dates of every detector\nRed bands: calendar events +-10 trading days; orange dotted: post-hoc extra events", fontsize=9)
fig.tight_layout(); fig.savefig(P.FIGS/"fig2_alarms.png"); plt.close()

# ---------------- Fig 3: real-data metrics ------------------------------------------------------------
sc = R["score"][10]; ch = R["chance"]
nm3 = [x for x in names]
fig, ax = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
y = np.arange(len(nm3))
cols = [C["blue"] if j < 5 else (C["grey"] if j < 10 else C["purple"]) for j in range(len(nm3))]
ax[0].barh(y, [sc[k]["recall"] for k in nm3], color=cols)
ax[0].scatter([ch[k]["exp_recall"] for k in nm3], y, color="k", marker="D", s=14, zorder=3, label="random alarms (same count)")
ax[0].set_title("Recall (5 events, +-10 d)"); ax[0].legend(fontsize=7, loc="lower right")
ax[1].barh(y, [sc[k]["precision"] for k in nm3], color=cols)
ax[1].scatter([ch[k]["exp_precision"] for k in nm3], y, color="k", marker="D", s=14, zorder=3)
ax[1].set_title("Precision")
ax[2].barh(y, [sc[k]["far"] for k in nm3], color=cols); ax[2].set_title("False alarms per year")
ax[0].set_yticks(y); ax[0].set_yticklabels(nm3, fontsize=8); ax[0].invert_yaxis()
fig.suptitle("Figure 3. Real S&P 500: detection quality against the frozen calendar", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig3_real_metrics.png"); plt.close()

# ---------------- Fig 4: synthetic detection by condition -------------------------------------------
cr, cd = O["cond_rate"], O["cond_delay"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, M, ttl, cm, fmt in ((ax[0], cr, "Detection rate within 125 days", "Blues", "{:.2f}"),
                           (ax[1], cd, "Median delay (days, detected series)", "Oranges_r", "{:.0f}")):
    im = a.imshow(M.values, cmap=cm, aspect="auto")
    a.set_xticks(range(M.shape[1])); a.set_xticklabels([c.replace(" (", "\n(") for c in M.columns], rotation=70, ha="right", fontsize=7)
    a.set_yticks(range(M.shape[0])); a.set_yticklabels(M.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                a.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5, color="k")
    a.set_title(ttl)
fig.suptitle("Figure 4. Synthetic benchmark (25 series per condition, known drift date)", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig4_synth_cond.png"); plt.close()

# ---------------- Fig 5: operating points vs chance ----------------------------------------------------
E1 = O["E1"]; ab = O["abl_synth"]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
xs = np.linspace(0, 6, 100)
ax.plot(xs, 1 - np.exp(-xs * 125 / 252), color="k", lw=1, ls="--", label="chance (random alarms, same FAR)")
for d in E1.index:
    c = C["blue"] if d in ("KS/MMD (features)", "ADWIN (forecast error)", "Page-Hinkley (forecast error)",
                           "BOCPD (target)", "Fusion (>=2 of 3 families)") else C["grey"]
    ax.scatter(E1.loc[d, "far"], E1.loc[d, "det_rate"], color=c, s=40, zorder=3)
    ax.annotate(d.split(" (")[0], (E1.loc[d, "far"], E1.loc[d, "det_rate"]), fontsize=7, xytext=(4, -9), textcoords="offset points")
tun = [r for r in ab.index if r.startswith(("Page-Hinkley lambda", "BOCPD threshold", "ADWIN delta", "Fusion any", "Fusion all"))]
ax.scatter(ab.loc[tun, "far"], ab.loc[tun, "det_rate"], facecolors="none", edgecolors=C["orange"], s=40, label="threshold variants")
ax.set_xlabel("False alarms per year"); ax.set_ylabel("Detection rate (<=125 days)")
ax.set_title("Figure 5. Detection rate vs false-alarm rate (synthetic)", fontsize=10)
ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(P.FIGS/"fig5_operating.png"); plt.close()

# ---------------- Fig 6: Friedman mean ranks ---------------------------------------------------------
fr = O["friedman"]
rk = pd.Series(fr["ranks"]).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(range(len(rk)), rk.values, color=C["blue"]); ax.set_yticks(range(len(rk))); ax.set_yticklabels(rk.index, fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("Mean rank of detection delay (1 = fastest, miss = 125 d)")
ax.errorbar(rk.values.max() + 0.2, len(rk) - 0.2, xerr=fr["CD"] / 2, color=C["red"], capsize=3)
ax.text(rk.values.max() + 0.2, len(rk) - 0.8, f"CD = {fr['CD']:.2f}", color=C["red"], fontsize=8, ha="center")
ax.set_title(f"Figure 6. Friedman test (chi2 = {fr['stat']:.0f}, p < 1e-90, N = {fr['N']} series); Nemenyi CD", fontsize=9)
fig.tight_layout(); fig.savefig(P.FIGS/"fig6_ranks.png"); plt.close()

# ---------------- Fig 7: E3a separation -----------------------------------------------------------------
sep = O["sep"]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(sep)); w = 0.38
ax.bar(x - w / 2, sep.auc_ks, w, color=C["purple"], label="covariate score (KS peak rise)")
ax.bar(x + w / 2, sep.auc_err, w, color=C["green"], label="error score (log error ratio)")
ax.axhline(0.5, color="k", lw=0.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels(sep.index, rotation=25, ha="right")
ax.set_ylabel("AUC vs. no-drift series"); ax.set_ylim(0.3, 1.05); ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_title("Figure 7. Which signal reacts to which drift type? (0.5 = no separation)", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig7_e3a.png"); plt.close()

# ---------------- Fig 8: E3b lead-lag -------------------------------------------------------------------
lags, ccf = R["ccf"]; g = R["gaps"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(lags, ccf, color=C["blue"], width=0.9); ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_xlabel("lag L (days): corr(drift score_t, error_{t+L}); L>0 = drift score leads")
ax[0].set_ylabel("cross-correlation"); ax[0].set_title("Cross-correlation of changes", fontsize=9)
for j, (fam, col, lab) in enumerate((("cov", C["purple"], "KS/MMD (covariate)"), ("err", C["green"], "ADWIN/PH (error)"),
                                     ("bocpd", C["orange"], "BOCPD"))):
    ax[1].scatter(g[fam], np.arange(len(g)) + (j - 1) * 0.15, color=col, s=36, label=lab)
ax[1].axvline(0, color="k", lw=0.8); ax[1].set_yticks(range(len(g))); ax[1].set_yticklabels([short[e] for e in g.event])
ax[1].invert_yaxis(); ax[1].set_xlabel("first alarm date minus date forecast error begins to rise (days)")
ax[1].legend(fontsize=7); ax[1].set_title("Alarm timing relative to error rise (>0: alarm later)", fontsize=9)
fig.suptitle("Figure 8. Real data (S&P 500): which moves first?", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig8_e3b.png"); plt.close()

# ---------------- Fig 9: COVID error distribution shift -------------------------------------------------
e = R["err"]; dts = dates
per = {"pre-COVID (2017-19)": ("2017-01-01", "2019-12-31"), "COVID (Feb-Dec 2020)": ("2020-02-20", "2020-12-31"),
       "post (2021-22)": ("2021-01-01", "2022-12-31")}
fig, ax = plt.subplots(figsize=(6, 3.8))
for (nm, (a, b)), col in zip(per.items(), (C["blue"], C["red"], C["green"])):
    v = np.sort(e[(dts >= a) & (dts <= b) & ~np.isnan(e)])
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=nm, color=col)
ax.set_xlim(0, 4); ax.set_xlabel("one-step absolute forecast error (% return)"); ax.set_ylabel("empirical CDF")
ax.legend(fontsize=8); ax.set_title("Figure 9. Forecast-error distribution across periods", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig9_covid.png"); plt.close()

# ---------------- Fig 10: ablations ----------------------------------------------------------------------
abl = O["abl_synth"]
groups = [("KS/MMD variants", ["Full KS/MMD (alpha=0.01)", "KS/MMD alpha=0.02", "KS/MMD alpha=0.05", "KS only (no MMD)",
                              "MMD only (no KS)", "KS/MMD without feature 1", "KS/MMD without feature 2",
                              "KS/MMD without feature 3", "KS/MMD without Holm correction"]),
          ("Fusion variants", ["Fusion full (>=2 of 3)", "Fusion any family (k=1)", "Fusion all three (k=3)",
                              "Fusion without cov monitor", "Fusion without err monitor", "Fusion without bocpd monitor",
                              "Fusion window 5 d", "Fusion window 21 d"]),
          ("Error monitors / BOCPD", ["ADWIN delta=0.02", "ADWIN delta=0.1", "ADWIN delta=0.3", "Page-Hinkley lambda=4.0",
                                      "Page-Hinkley lambda=16.0", "Page-Hinkley lambda=64.0", "BOCPD threshold=0.6",
                                      "BOCPD threshold=0.9", "BOCPD threshold=0.95"])]
fig, ax = plt.subplots(1, 3, figsize=(12, 4.6))
for a, (ttl, rows) in zip(ax, groups):
    yy = np.arange(len(rows))
    a.barh(yy - 0.2, abl.loc[rows, "det_rate"], 0.4, color=C["blue"], label="detection rate")
    a.barh(yy + 0.2, abl.loc[rows, "chance_det"], 0.4, color=C["grey"], label="chance level (same FAR)")
    for i, r_ in enumerate(rows):
        a.text(1.01, i, f"FAR {abl.loc[r_, 'far']:.2f}", fontsize=6.5, va="center")
    a.set_yticks(yy); a.set_yticklabels(rows, fontsize=7); a.invert_yaxis(); a.set_xlim(0, 1.25); a.set_title(ttl, fontsize=9)
ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Figure 10. Ablation on synthetic series (detection rate vs chance level; FAR = false alarms per year)", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig10_ablation.png"); plt.close()

# ---------------- Fig 11: leakage ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 3.6))
keys = list(LK.keys()); lab = ["Random split\n(An et al. style)", "Chronological +\n20-day purge", "Always warn\n(baseline)"]
f1 = [LK[k]["f1"] for k in keys]
ax.bar(lab, f1, color=[C["red"], C["blue"], C["grey"]])
for i, v in enumerate(f1):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
ax.set_ylabel("F-measure on held-out data"); ax.set_ylim(0, 1.05)
ax.set_title("Figure 11. Effect of the evaluation protocol (same data, same model)", fontsize=9)
fig.tight_layout(); fig.savefig(P.FIGS/"fig11_leakage.png"); plt.close()

# ---------------- Fig 12: dashboard (static capture) ------------------------------------------------------
cov = R["cov"]
fig, ax = plt.subplots(4, 1, figsize=(9.5, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.3, 0.8, 0.9]})
ax[0].plot(dates, R["close"].values, color=C["blue"], lw=0.9); ax[0].set_yscale("log"); ax[0].set_ylabel("S&P 500")
fam = R["fam"]
for fk, col, row in (("cov", C["purple"], 0), ("err", C["green"], 1), ("bocpd", C["orange"], 2)):
    ax[0].scatter([dates[a] for a in fam[fk]], [R["close"].min() * 0.7 * (1.25 ** row)] * len(fam[fk]), marker="^", s=16, color=col)
H_ = np.array([cov["stats"][:, j] for j in range(6)])
ax[1].imshow(H_, aspect="auto", cmap="magma", extent=[mdates.date2num(dates[cov["days"][0]]), mdates.date2num(dates[cov["days"][-1]]), 6, 0])
ax[1].xaxis_date(); ax[1].set_yticks(np.arange(6) + 0.5); ax[1].set_yticklabels(D.FEATURES, fontsize=7)
ax[1].set_ylabel("feature distribution\nshift (KS stat)")
agree = np.zeros(n)
for fk in ("cov", "err", "bocpd"):
    f_ = np.zeros(n)
    for a in fam[fk]:
        f_[a:a + 11] = 1
    agree += f_
ax[2].fill_between(dates, agree, step="mid", color=C["red"], alpha=0.7); ax[2].set_ylabel("monitors in\nalarm (of 3)"); ax[2].set_yticks([0, 1, 2, 3])
ax[3].plot(dates, R["bocpd_pcp"], color=C["orange"], lw=0.7, label="BOCPD P(change in last 10 d)")
p = np.full(n, np.nan); p[cov["days"]] = -np.log10(cov["p"].min(1))
ax[3].plot(dates, p / 3.0, color=C["purple"], lw=0.7, label="KS/MMD evidence (-log10 p / 3)")
ax[3].set_ylim(0, 1.05); ax[3].set_ylabel("uncertainty / evidence"); ax[3].legend(fontsize=7, loc="upper left", ncol=2)
for a in ax:
    evlines(a)
fig.suptitle("Figure 12. Drift dashboard (static capture): data, feature timeline, agreement, uncertainty", fontsize=10)
fig.tight_layout(); fig.savefig(P.FIGS/"fig12_dashboard.png"); plt.close()
print("figures done", sorted(os.listdir(P.FIGS)))
