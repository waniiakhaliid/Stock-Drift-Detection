"""analyze.py - metrics, Friedman/Nemenyi, E3a attribution, ablations. Writes tables.pkl"""
import pickle, time
import paths as P
import numpy as np, pandas as pd
from scipy import stats as st
import scikit_posthocs as sp
import drift as D

H = 125                      # detection horizon after a synthetic drift (trading days)
t0 = time.time()
S = pickle.load(open(P.RESULTS/"synth_results.pkl", "rb"))
R = pickle.load(open(P.RESULTS/"results_SP500.pkl", "rb"))
series, NS, T = S["series"], S["N_SETUP"], S["T"]
DET = ["KS/MMD (features)", "ADWIN (forecast error)", "Page-Hinkley (forecast error)", "BOCPD (target)",
       "Fusion (>=2 of 3 families)", "HMM (online, 2-state)", "Jump model (simplified)", "Volatility rule",
       "Fixed schedule (252 d)", "ADWIN on |y| (Neri-style)"]
OUT = {}


def synth_metrics(alarm_fn, subset=None):
    """alarm_fn(series_dict)-> list of alarm days. Returns per-series records."""
    rec = []
    for s in series:
        if subset is not None and not subset(s):
            continue
        a = alarm_fn(s)
        if s["tau"] is None:
            rec.append(dict(kind="null", spec=None, fa=len(a), yrs=(T - NS) / 252, delay=np.nan, det=np.nan))
        else:
            tau = s["tau"]
            pre = [x for x in a if x < tau]
            post = [x for x in a if tau <= x <= tau + H]
            rec.append(dict(kind=s["kind"], spec=s["spec"], fa=len(pre), yrs=(tau - NS) / 252,
                            delay=(min(post) - tau) if post else np.nan, det=1.0 if post else 0.0))
    return pd.DataFrame(rec)


def summarize(df):
    dr = df[df.kind != "null"]
    far = df.fa.sum() / df.yrs.sum()
    chance = 1 - np.exp(-far * H / 252.0)      # detection rate a random alarm process with the same FAR would get
    return dict(far=far, det_rate=dr.det.mean(), chance_det=chance, excess=dr.det.mean() - chance,
                med_delay=dr.delay.median(), mean_delay_cap=dr.delay.fillna(H).mean())


# ---- E1: detector benchmark ------------------------------------------------------------------
tab, per_series = {}, {}
for d in DET:
    df = synth_metrics(lambda s, d=d: s["alarms"][d])
    per_series[d] = df
    tab[d] = summarize(df)
E1 = pd.DataFrame(tab).T
OUT["E1"] = E1
print(E1.round(3))

# by condition
specs = list(D.SPECS)
cond_rate = pd.DataFrame({d: {sp_: per_series[d][per_series[d].spec == sp_].det.mean() for sp_ in specs} for d in DET})
cond_delay = pd.DataFrame({d: {sp_: per_series[d][per_series[d].spec == sp_].delay.median() for sp_ in specs} for d in DET})
OUT["cond_rate"], OUT["cond_delay"] = cond_rate, cond_delay

# Friedman + Nemenyi on delay (miss = H) across all drifted series
M = np.column_stack([per_series[d][per_series[d].kind != "null"].delay.fillna(H).values for d in DET])
fr = st.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
ranks = np.apply_along_axis(st.rankdata, 1, M).mean(0)
nem = sp.posthoc_nemenyi_friedman(M)
nem.columns = nem.index = DET
k, N = M.shape[1], M.shape[0]
q = {5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219}[k]
CD = q * np.sqrt(k * (k + 1) / (6 * N))
OUT["friedman"] = dict(stat=fr.statistic, p=fr.pvalue, ranks=dict(zip(DET, ranks)), CD=CD, N=N, k=k)
OUT["nemenyi"] = nem
print("Friedman", fr, "CD", CD); print(dict(zip(DET, ranks.round(2))))

# ---- E3a: covariate vs concept (does the right monitor fire?) -----------------------------------
fam_rows = []
for s in series:
    if s["tau"] is None:
        continue
    row = dict(kind=s["kind"], spec=s["spec"])
    for f_, al in s["fam"].items():
        row[f_] = float(any(s["tau"] <= x <= s["tau"] + H for x in al))
        row[f_ + "_pre"] = len([x for x in al if x < s["tau"]])
    fam_rows.append(row)
FR = pd.DataFrame(fam_rows)
OUT["fam_fire"] = FR.groupby("kind")[["cov", "err", "bocpd"]].mean()
OUT["fam_fire_spec"] = FR.groupby("spec")[["cov", "err", "bocpd"]].mean().loc[specs]


def attribute(r):
    if r["cov"] and not r["err"]:
        return "covariate"
    if r["err"] and not r["cov"]:
        return "concept"
    if r["err"] and r["cov"]:
        return "joint"
    return "undetected"


FR["pred"] = FR.apply(attribute, axis=1)
conf = pd.crosstab(FR.kind, FR.pred).reindex(index=["covariate", "concept", "joint"],
                                                 columns=["covariate", "concept", "joint", "undetected"], fill_value=0)
OUT["confusion"] = conf
cc = FR[FR.pred != "undetected"]
OUT["attr_acc_all"] = float((FR.kind == FR.pred).mean())
OUT["attr_acc_detected"] = float((cc.kind == cc.pred).mean())
print(OUT["fam_fire"].round(2)); print(conf)


# ---- E3a (continuous, alarm-free): do the scores separate covariate from concept drift? -------------
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(st.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))


def scores(s):
    tau = s["tau"] if s["tau"] is not None else 1000 + s["seed"] % 301      # pseudo-tau for null series
    days, ks = s["cov_days"], s["cov_stats"][:, :3].max(1)
    pre = ks[(days >= tau - 126) & (days < tau)].max()
    post = ks[(days >= tau) & (days <= tau + 126)].max()
    e = s["e"]
    er = np.log(np.nanmean(e[tau:tau + 126]) / np.nanmean(e[tau - 126:tau]))
    return post - pre, er


sc = pd.DataFrame([dict(spec=s["spec"], kind=s["kind"], **dict(zip(["d_ks", "log_err"], scores(s)))) for s in series])
nul = sc[sc.kind == "null"]
rows = []
for sp_ in specs:
    g = sc[sc.spec == sp_]
    rows.append(dict(spec=sp_, kind=D.KIND[sp_], med_d_ks=g.d_ks.median(), med_log_err=g.log_err.median(),
                     auc_ks=auc(g.d_ks, nul.d_ks), auc_err=auc(g.log_err, nul.log_err)))
OUT["sep"] = pd.DataFrame(rows).set_index("spec")
OUT["sep_null"] = dict(d_ks=nul.d_ks.median(), log_err=nul.log_err.median())
print(OUT["sep"].round(3))
# null firing rate of each family in a pseudo window (alarm-rate chance level)
nf = {}
for f_ in ("cov", "err", "bocpd"):
    v = []
    for s in series:
        if s["tau"] is None:
            tau = 1000 + s["seed"] % 301
            v.append(float(any(tau <= x <= tau + H for x in s["fam"][f_])))
    nf[f_] = float(np.mean(v))
OUT["null_fire"] = nf
print("null firing", nf)

# ---- synthetic ablations -------------------------------------------------------------------------
def fuse(fam, keys, win=10, kk=2):
    out = []
    pts = sorted([(a, key) for key in keys for a in fam[key]])
    for a, key in pts:
        ag = {k2 for k2 in keys if any(0 <= a - b <= win for b in fam[k2])}
        if len(ag) >= kk and (not out or a - out[-1] >= 21):
            out.append(a)
    return out


abl = {}


def add(name, fn):
    abl[name] = summarize(synth_metrics(fn))


def cov_alarms(s, alpha=0.01, use=None):
    flags = D.holm_reject(s["cov_p"], alpha, use)
    full = np.zeros(T, int); full[s["cov_days"][flags]] = 1
    return D.episodes(full, 21, NS)


add("Full KS/MMD (alpha=0.01)", cov_alarms)
for a_ in (0.02, 0.05):
    add(f"KS/MMD alpha={a_}", lambda s, a_=a_: cov_alarms(s, a_))
add("KS only (no MMD)", lambda s: cov_alarms(s, 0.01, [0, 1, 2]))
add("MMD only (no KS)", lambda s: cov_alarms(s, 0.01, [3]))
for j in range(3):
    add(f"KS/MMD without feature {j + 1}", lambda s, j=j: cov_alarms(s, 0.01, [c for c in range(4) if c != j]))
# no Holm correction (reject if any raw p < alpha)
def cov_nocorr(s):
    flags = (s["cov_p"] < 0.01).any(1)
    full = np.zeros(T, int); full[s["cov_days"][flags]] = 1
    return D.episodes(full, 21, NS)
add("KS/MMD without Holm correction", cov_nocorr)
# fusion variants
K3 = ["cov", "err", "bocpd"]
add("Fusion full (>=2 of 3)", lambda s: s["alarms"]["Fusion (>=2 of 3 families)"])
add("Fusion any family (k=1)", lambda s: fuse(s["fam"], K3, 10, 1))
add("Fusion all three (k=3)", lambda s: fuse(s["fam"], K3, 10, 3))
for drop in K3:
    keys = [x for x in K3 if x != drop]
    add(f"Fusion without {drop} monitor", lambda s, keys=keys: fuse(s["fam"], keys, 10, 2))
add("Fusion window 5 d", lambda s: fuse(s["fam"], K3, 5, 2))
add("Fusion window 21 d", lambda s: fuse(s["fam"], K3, 21, 2))


# error monitor / BOCPD sensitivity (re-run streams on stored error series)
def re_adwin(delta):
    cache = {}
    def f(s):
        k = id(s)
        if k not in cache:
            cache[k] = D.episodes(D.run_adwin(s["en"], delta), 21, NS)
        return cache[k]
    return f


for dl in (0.02, 0.1, 0.3):
    add(f"ADWIN delta={dl}", re_adwin(dl))
for lam in (4.0, 16.0, 64.0):
    add(f"Page-Hinkley lambda={lam}", lambda s, lam=lam: D.episodes(D.run_ph(s["e"], lam), 21, NS))
for th in (0.6, 0.9, 0.95):
    add(f"BOCPD threshold={th}", lambda s, th=th: D.episodes(D.cross_up(s["pcp"], th), 21, NS))
OUT["abl_synth"] = pd.DataFrame(abl).T
print(OUT["abl_synth"].round(3))

# window-size ablation (re-runs the monitor; subset of series)
sub_specs = {"cov_mean_1", "cov_var_2", "concept_flip", "joint"}
sub = [s for s in series if s["spec"] in sub_specs and (s["seed"] % 2 == 0)] + \
      [s for s in series if s["spec"] is None][:15]
winres = {}
for W in (21, 63, 126):
    rows = []
    for s in sub:
        spec, tau = s["spec"], s["tau"]
        X, y, _ = D.simulate(spec, T, tau if tau else 0, s["seed"])
        mu, sd = X[:NS].mean(0), X[:NS].std(0)
        Z = (X - mu) / sd
        if W == 63:
            res = dict(days=s["cov_days"], p=s["cov_p"])
        else:
            res = D.cov_monitor(Z, NS, 252, W, B=1000, stride=5, seed=s["seed"])
        flags = D.holm_reject(res["p"], 0.01)
        full = np.zeros(T, int); full[res["days"][flags]] = 1
        a = D.episodes(full, 21, NS)
        if tau is None:
            rows.append(dict(kind="null", spec=None, fa=len(a), yrs=(T - NS) / 252, delay=np.nan, det=np.nan))
        else:
            pre = [x for x in a if x < tau]; post = [x for x in a if tau <= x <= tau + H]
            rows.append(dict(kind=s["kind"], spec=spec, fa=len(pre), yrs=(tau - NS) / 252,
                             delay=(min(post) - tau) if post else np.nan, det=1.0 if post else 0.0))
    winres[f"test window W={W}"] = summarize(pd.DataFrame(rows))
OUT["abl_window"] = pd.DataFrame(winres).T
print(OUT["abl_window"].round(3), flush=True)

# ---- real-data ablations -------------------------------------------------------------------------
n, ns = R["n"], R["n_setup"]
evl = sorted(R["events"].values())
cov = R["cov"]
e = R["err"]; q99 = R["params"]["q99"]; en = np.minimum(e / q99, 1.0)
pcp = R["bocpd_pcp"]
ra = {}


def rscore(alarms, tol=10):
    m = D.evaluate_events(alarms, evl, n, ns, tol)
    return dict(n_alarms=m["n_alarms"], precision=m["precision"], recall=m["recall"], far=m["far"],
                median_delay=m["median_delay"])


def rcov(alpha=0.01, use=None):
    return D.episodes(D.cov_alarm_days(cov, alpha, use, n), 21, ns)


ra["Full KS/MMD (alpha=0.01)"] = rscore(rcov())
for a_ in (0.02, 0.05):
    ra[f"KS/MMD alpha={a_}"] = rscore(rcov(a_))
ra["KS only (no MMD)"] = rscore(rcov(0.01, list(range(6))))
ra["MMD only (no KS)"] = rscore(rcov(0.01, [6]))
for j, nm in enumerate(D.FEATURES):
    ra[f"KS/MMD without {nm}"] = rscore(rcov(0.01, [c for c in range(7) if c != j]))
ra["KS/MMD volatility features only"] = rscore(rcov(0.01, [1, 2, 3]))
for W in (21, 126):
    rr = D.cov_monitor(R["Z"], ns, 252, W, B=1000, stride=1)
    fl = D.cov_alarm_days(rr, 0.01, None, n)
    ra[f"KS/MMD test window W={W}"] = rscore(D.episodes(fl, 21, ns))
# no Holm
fl = np.zeros(n, int); fl[cov["days"][(cov["p"] < 0.01).any(1)]] = 1
ra["KS/MMD without Holm correction"] = rscore(D.episodes(fl, 21, ns))
fam = R["fam"]
ra["Fusion full (>=2 of 3)"] = rscore(R["alarms"]["Fusion (>=2 of 3 families)"])
ra["Fusion any family (k=1)"] = rscore(fuse(fam, K3, 10, 1))
ra["Fusion all three (k=3)"] = rscore(fuse(fam, K3, 10, 3))
for drop in K3:
    ra[f"Fusion without {drop} monitor"] = rscore(fuse(fam, [x for x in K3 if x != drop], 10, 2))
for dl in (0.02, 0.1, 0.3):
    ra[f"ADWIN delta={dl}"] = rscore(D.episodes(D.run_adwin(en, dl), 21, ns))
for lam in (4.0, 8.0, 16.0):
    ra[f"Page-Hinkley lambda={lam}"] = rscore(D.episodes(D.run_ph(e, lam), 21, ns))
for th in (0.2, 0.6, 0.9):
    ra[f"BOCPD threshold={th}"] = rscore(D.episodes(D.cross_up(pcp, th), 21, ns))
OUT["abl_real"] = pd.DataFrame(ra).T
print(OUT["abl_real"].round(3))

# tolerance sensitivity on real
tol_tab = {}
for tol in (5, 10, 20):
    for k_, v in R["score"][tol].items():
        tol_tab.setdefault(k_, {})[f"recall@{tol}"] = v["recall"]
        tol_tab[k_][f"precision@{tol}"] = v["precision"]
OUT["tol"] = pd.DataFrame(tol_tab).T

pickle.dump(OUT, open(P.RESULTS/"tables.pkl", "wb"))
print("analysis done", round(time.time() - t0))
