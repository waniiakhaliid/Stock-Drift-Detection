"""run_synth.py - E1 / E3a: synthetic series with KNOWN drift dates and known drift type.
Detectors are calibrated ONCE on a separate batch of null series (setup periods only), then frozen."""
import sys, pickle, time
import paths as P
import numpy as np
import drift as D

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 25
N_NULL = 30
T, N_SETUP = 1800, 500
t0 = time.time()


def prep(X, y):
    mu, sd = X[:N_SETUP].mean(0), X[:N_SETUP].std(0)
    Z = (X - mu) / sd
    ys = y / y[:N_SETUP].std()
    e, w = D.forecast_errors(Z, ys, N_SETUP)
    q99 = np.nanpercentile(e[:N_SETUP], 99)
    return Z, ys, e, np.minimum(e / q99, 1.0), q99


# ---- calibration batch (null series, setup periods only) --------------------------------------
rng = np.random.default_rng(7)
far = {"adwin": [], "ph": [], "bocpd": [], "adwin_raw": []}
for i in range(20):
    X, y, _ = D.simulate(None, T, 0, seed=9000 + i)
    Z, ys, e, en, q99 = prep(X, y)
    rn = np.minimum(np.abs(ys) / np.percentile(np.abs(ys[:N_SETUP]), 99), 1.0)
    far["adwin"].append(D.calibrate_stream(en[1:N_SETUP], "adwin", D.ADWIN_GRID, rng, B=15)[1])
    far["ph"].append(D.calibrate_stream(e[1:N_SETUP], "ph", D.PH_GRID, rng, B=15)[1])
    far["bocpd"].append(D.calibrate_stream(ys[:N_SETUP], "bocpd", D.BOCPD_GRID, rng, B=15, scale0=1.0)[1])
    far["adwin_raw"].append(D.calibrate_stream(rn[:N_SETUP], "adwin", D.ADWIN_GRID, rng, B=15)[1])


def pick(key, grid):
    pooled = {g: float(np.mean([f[g] for f in far[key]])) for g in grid}
    ok = [g for g in grid if pooled[g] <= 1.0]
    return (ok[-1] if ok else grid[0]), pooled


PARAMS = {}
PARAMS["adwin"], cal_a = pick("adwin", D.ADWIN_GRID)
PARAMS["ph"], cal_p = pick("ph", D.PH_GRID)
PARAMS["bocpd"], cal_b = pick("bocpd", D.BOCPD_GRID)
PARAMS["adwin_raw"], cal_r = pick("adwin_raw", D.ADWIN_GRID)
print("calibrated", PARAMS, round(time.time() - t0), flush=True)

# ---- the test series ---------------------------------------------------------------------------
jobs = [(s, k, 1000 + 97 * r_ + 13 * k) for k, s in enumerate(D.SPECS) for r_ in range(N_REP)]
jobs += [(None, 99, 5000 + j) for j in range(N_NULL)]
import os
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
out = []
if os.path.exists(P.RESULTS/"synth_results.pkl"):
    out = pickle.load(open(P.RESULTS/"synth_results.pkl", "rb"))["series"]
print("resuming from", len(out), flush=True)
for jn, (spec, k, seed) in enumerate(jobs):
    if jn < len(out):
        continue
    if time.time() - t0 > BUDGET:
        break
    tau = int(np.random.default_rng(seed).integers(1000, 1301))
    X, y, tau_ = D.simulate(spec, T, tau, seed)
    Z, ys, e, en, q99 = prep(X, y)
    rn = np.minimum(np.abs(ys) / np.percentile(np.abs(ys[:N_SETUP]), 99), 1.0)
    cov = D.cov_monitor(Z, N_SETUP, 252, 63, B=1000, stride=5, seed=seed)
    flags = {}
    flags["KS/MMD (features)"] = D.cov_alarm_days(cov, 0.01, n_total=T)
    flags["ADWIN (forecast error)"] = D.run_adwin(en, PARAMS["adwin"])
    flags["Page-Hinkley (forecast error)"] = D.run_ph(e, PARAMS["ph"])
    pcp = D.bocpd_pcp(ys, 1.0)
    flags["BOCPD (target)"] = D.cross_up(pcp, PARAMS["bocpd"])
    flags["HMM (online, 2-state)"] = D.hmm_flags(Z, N_SETUP)
    flags["Jump model (simplified)"] = D.jump_flags(Z, N_SETUP)
    import pandas as pd
    vol = pd.Series(ys).rolling(21).std().bfill().values
    flags["Volatility rule"] = D.vol_rule_flags(vol, N_SETUP)
    flags["Fixed schedule (252 d)"] = D.schedule_flags(T, N_SETUP)
    flags["ADWIN on |y| (Neri-style)"] = D.run_adwin(rn, PARAMS["adwin_raw"])
    al = {kk: D.episodes(v, 21, start=N_SETUP) for kk, v in flags.items()}
    fam = {"cov": al["KS/MMD (features)"],
           "err": sorted(al["ADWIN (forecast error)"] + al["Page-Hinkley (forecast error)"]),
           "bocpd": al["BOCPD (target)"]}
    pts = sorted([(a, kk) for kk in fam for a in fam[kk]]); fu = []
    for a, kk in pts:
        ag = {k2 for k2 in fam if any(0 <= a - b <= 10 for b in fam[k2])}
        if len(ag) >= 2 and (not fu or a - fu[-1] >= 21):
            fu.append(a)
    al["Fusion (>=2 of 3 families)"] = fu
    out.append(dict(spec=spec, kind=("null" if spec is None else D.KIND[spec]), tau=tau_, seed=seed, alarms=al,
                    fam=fam, cov_stats=cov["stats"], cov_p=cov["p"], cov_days=cov["days"],
                    e=e, en=en, pcp=pcp, ys=ys))
    if jn % 10 == 0:
        print(jn, "/", len(jobs), round(time.time() - t0), "s", flush=True)
        pickle.dump(dict(params=PARAMS, cal=(cal_a, cal_p, cal_b, cal_r), series=out, N_SETUP=N_SETUP, T=T),
                    open(P.RESULTS/"synth_results.pkl", "wb"))
pickle.dump(dict(params=PARAMS, cal=(cal_a, cal_p, cal_b, cal_r), series=out, N_SETUP=N_SETUP, T=T),
            open(P.RESULTS/"synth_results.pkl", "wb"))
print("done", round(time.time() - t0), flush=True)