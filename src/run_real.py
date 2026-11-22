"""run_real.py - E2 / E3b: all detectors on a real index, scored against a frozen event calendar.
Usage: python run_real.py                      (S&P 500, data/SP500_close.csv)
       python run_real.py --csv KSE100.csv --name KSE100 --events events_KSE100.csv   (files looked up in data/)"""
import sys, pickle, time, argparse
import paths as P
import numpy as np, pandas as pd
import drift as D

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=None)
ap.add_argument("--name", default="SP500")
ap.add_argument("--events", default=None, help="optional CSV with columns date,label for another index")
args = ap.parse_args()

# ---- frozen event calendar (written BEFORE any detector was run) -----------------------------
EVENTS = {  # onset-of-stress dates, S&P 500
    "2008 Lehman collapse": "2008-09-15",
    "2011 US downgrade selloff": "2011-08-08",
    "Dec 2018 selloff": "2018-12-04",
    "2020 COVID crash onset": "2020-02-24",
    "2022 rate-hike selloff": "2022-01-05",
}
# post-hoc sensitivity calendar (added AFTER first results; reported separately, never used for tuning)
EXTRA = {
    "2010 flash crash": "2010-05-06",
    "2015 China devaluation": "2015-08-24",
    "2016 Brexit vote": "2016-06-24",
    "2018 Feb volatility spike": "2018-02-05",
    "2022 June CPI shock": "2022-06-13",
}
if args.events:
    e = pd.read_csv(P.data_file(args.events))
    EVENTS = dict(zip(e["label"], e["date"])); EXTRA = {}

t0 = time.time()
if args.csv:
    close = D.load_csv_index(P.data_file(args.csv))
elif (P.DATA / "SP500_close.csv").exists():      # offline copy of the skfolio S&P 500 series
    close = D.load_csv_index(P.DATA / "SP500_close.csv")
else:
    close = D.load_sp500()                             # needs: pip install skfolio
F_all = D.make_features(close)
F = F_all[F_all.index >= "2005-01-01"].copy()
setup_end = F.index[0] + pd.DateOffset(years=2)
n_setup = int((F.index < setup_end).sum())
Z_df, mu, sd = D.scale_on_setup(F, n_setup)
Z = Z_df.values
n = len(Z)
r = F["ret"].values                      # percent log return
y = np.abs(r)                            # forecast target: |return| (one-day proxy for volatility)
print(args.name, "days", n, "setup", n_setup, F.index[0].date(), "->", F.index[-1].date())

def idx(date):
    return int(np.searchsorted(F.index, pd.Timestamp(date)))
ev = {k: idx(v) for k, v in EVENTS.items()}
ex = {k: idx(v) for k, v in EXTRA.items()}

R = {"name": args.name, "dates": F.index, "n": n, "n_setup": n_setup, "events": ev, "extra": ex,
     "close": close.reindex(F.index), "F": F, "Z": Z}

# ---- 5A covariate monitor --------------------------------------------------------------------
cov = D.cov_monitor(Z, n_setup, R=252, W=63, B=1000, stride=1)
R["cov"] = cov
print("cov monitor done", round(time.time() - t0))

# ---- forecaster + error monitors -------------------------------------------------------------
e, w = D.forecast_errors(Z, y, n_setup)
R["err"] = e
q99 = np.nanpercentile(e[:n_setup], 99)
en = np.minimum(e / q99, 1.0)
rng = np.random.default_rng(D.RNG_SEED)
cal = {}
d_adwin, cal["adwin"] = D.calibrate_stream(en[1:n_setup], "adwin", D.ADWIN_GRID, rng)
d_ph, cal["ph"] = D.calibrate_stream(e[1:n_setup], "ph", D.PH_GRID, rng)
th_bocpd, cal["bocpd"] = D.calibrate_stream(r[:n_setup], "bocpd", D.BOCPD_GRID, rng, scale0=np.std(r[:n_setup]))
rn = np.minimum(y / np.percentile(y[:n_setup], 99), 1.0)
d_raw, cal["adwin_raw"] = D.calibrate_stream(rn[:n_setup], "adwin", D.ADWIN_GRID, rng)
R["cal"] = cal
R["params"] = dict(adwin_delta=d_adwin, ph_lambda=d_ph, bocpd_thr=th_bocpd, adwin_raw_delta=d_raw, q99=q99,
                   block=D.block_len(r[:n_setup]), ridge_w=w)
print("params", R["params"], round(time.time() - t0))

flags = {}
flags["ADWIN (forecast error)"] = D.run_adwin(en, d_adwin)
flags["Page-Hinkley (forecast error)"] = D.run_ph(e, d_ph)
pcp = D.bocpd_pcp(r, np.std(r[:n_setup]))
R["bocpd_pcp"] = pcp
flags["BOCPD (returns)"] = D.cross_up(pcp, th_bocpd)
flags["KS/MMD (features)"] = D.cov_alarm_days(cov, 0.01, n_total=n)
print("proposed monitors done", round(time.time() - t0))

# ---- baselines -------------------------------------------------------------------------------
cols = [F.columns.get_loc("ret"), F.columns.get_loc("vol21")]
flags["HMM (online, 2-state)"] = D.hmm_flags(Z[:, cols], n_setup)
jc = [F.columns.get_loc(c) for c in ("vol21", "dd21", "ret")]
flags["Jump model (simplified)"] = D.jump_flags(Z[:, jc], n_setup)
flags["Volatility rule"] = D.vol_rule_flags(F["vol21"].values, n_setup)
flags["Fixed schedule (252 d)"] = D.schedule_flags(n, n_setup)
flags["ADWIN on |returns| (Neri-style)"] = D.run_adwin(rn, d_raw)
print("baselines done", round(time.time() - t0))

# ---- alarm episodes, fusion -------------------------------------------------------------------
alarms = {k: D.episodes(v, 21, start=n_setup) for k, v in flags.items()}
fam = {"cov": alarms["KS/MMD (features)"],
       "err": sorted(alarms["ADWIN (forecast error)"] + alarms["Page-Hinkley (forecast error)"]),
       "bocpd": alarms["BOCPD (returns)"]}
def fuse(fam, keys, win=10, k=2):
    out = []
    pts = sorted([(a, key) for key in keys for a in fam[key]])
    for a, key in pts:
        agree = {kk for kk in keys if any(0 <= a - b <= win for b in fam[kk])}
        if len(agree) >= k and (not out or a - out[-1] >= 21):
            out.append(a)
    return out
alarms["Fusion (>=2 of 3 families)"] = fuse(fam, ["cov", "err", "bocpd"])
R["flags"], R["alarms"], R["fam"] = flags, alarms, fam

# ---- offline hindsight reference (PELT on returns, variance changes) ---------------------------
import ruptures as rpt
algo = rpt.Pelt(model="normal", min_size=21, jump=5).fit(r.reshape(-1, 1))
cps = [c for c in algo.predict(pen=8 * np.log(n)) if c < n]
R["pelt_cps"] = cps
alarms["PELT offline (hindsight ceiling)"] = [c for c in cps if c >= n_setup]

# ---- scoring -----------------------------------------------------------------------------------
evl = sorted(ev.values())
exl = sorted(list(ev.values()) + list(ex.values()))
score = {}
for tol in (5, 10, 20):
    score[tol] = {k: D.evaluate_events(a, evl, n, n_setup, tol) for k, a in alarms.items()}
R["score"] = score
chance = {}
for k, a in alarms.items():
    rec, prec = D.chance_reference(len(a), evl, n, n_setup, 10)
    ob = score[10][k]
    chance[k] = dict(exp_recall=rec.mean(), p_recall=float((rec >= ob["recall"] - 1e-9).mean()),
                     exp_precision=prec.mean())
R["chance"] = chance
R["score_extended"] = {k: D.evaluate_events(a, exl, n, n_setup, 10) for k, a in alarms.items()}
R["score_late"] = {k: D.evaluate_events(a, evl, n, n_setup, 10) for k, a in alarms.items()}
print(pd.DataFrame({k: {m: v[m] for m in ("n_alarms", "precision", "recall", "far", "median_delay", "lead")}
                    for k, v in score[10].items()}).T.round(2))

# ---- E3b: which moves first? -----------------------------------------------------------------
days = cov["days"]
ks_score = np.full(n, np.nan); ks_score[days] = cov["stats"][:, :-1].max(1)
mmd_score = np.full(n, np.nan); mmd_score[days] = cov["stats"][:, -1]
err_roll = pd.Series(e).rolling(21).mean().values
R["ks_score"], R["mmd_score"], R["err_roll"] = ks_score, mmd_score, err_roll

def err_rise(i, half=60):
    lo, hi = max(n_setup + 10, i - half), min(n - 10, i + half)
    seg = e[lo:hi]; best, arg = -1, None
    for k in range(10, len(seg) - 10):
        a, b = seg[:k], seg[k:]
        t = (b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + 1e-12)
        if t > best:
            best, arg = t, lo + k
    return arg
rise = {k: err_rise(i) for k, i in ev.items()}
gaps = []
for k, i in ev.items():
    row = dict(event=k, event_idx=i, err_rise=rise[k])
    for fam_name, al in fam.items():
        w_ = [a for a in al if abs(a - i) <= 60]
        row[fam_name] = (min(w_) - rise[k]) if (w_ and rise[k] is not None) else np.nan   # rise is None when the error never rises (e.g. frozen prices)
    gaps.append(row)
R["gaps"] = pd.DataFrame(gaps)

from statsmodels.tsa.stattools import grangercausalitytests
a = pd.Series(ks_score).diff().rolling(5).mean(); b = pd.Series(err_roll).diff().rolling(5).mean()
dfc = pd.concat([a, b], axis=1).dropna().iloc[n_setup:]
dfc.columns = ["ks", "err"]
lags = list(range(-40, 41))
ccf = [dfc["ks"].corr(dfc["err"].shift(-L)) for L in lags]   # corr(ks_t, err_{t+L}); L>0 => ks leads
R["ccf"] = (lags, ccf)
gr = {}
for ml in (5, 10, 21):
    g1 = grangercausalitytests(dfc[["err", "ks"]], ml)[ml][0]["ssr_ftest"][1]   # ks -> err
    g2 = grangercausalitytests(dfc[["ks", "err"]], ml)[ml][0]["ssr_ftest"][1]   # err -> ks
    gr[ml] = (g1, g2)
R["granger"] = gr
print(R["gaps"]); print("granger", gr)
print("CCF peak lag", lags[int(np.nanargmax(ccf))], "max", np.nanmax(ccf))

# ---- Mirza-style error-distribution shift test across COVID periods ---------------------------
from scipy import stats as st
dts = F.index
def per(a, b):
    m = (dts >= a) & (dts <= b); return e[m & ~np.isnan(e)]
pre, cov_, post = per("2017-01-01", "2019-12-31"), per("2020-02-20", "2020-12-31"), per("2021-01-01", "2022-12-31")
tests = {}
for nm, (a, b) in {"pre vs COVID": (pre, cov_), "COVID vs post": (cov_, post), "pre vs post": (pre, post)}.items():
    tests[nm] = dict(ks=st.ks_2samp(a, b).statistic, ks_p=st.ks_2samp(a, b).pvalue,
                     lev=st.levene(a, b).statistic, lev_p=st.levene(a, b).pvalue)
R["mirza"] = dict(tests=tests, mae=dict(pre=pre.mean(), covid=cov_.mean(), post=post.mean()))
print(R["mirza"])

pickle.dump(R, open(P.RESULTS/f"results_{args.name}.pkl", "wb"))
print("saved", round(time.time() - t0), "s")
