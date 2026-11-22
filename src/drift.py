"""
drift.py - core library for "Watching the Distribution Move" (Phase 3)
All detectors are ONLINE: the value at day t uses data up to day t only.
Settings are calibrated on the setup period and then frozen.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from river.drift import ADWIN, PageHinkley

warnings.filterwarnings("ignore")
RNG_SEED = 42

# ----------------------------------------------------------------------------
# 1. Data and features
# ----------------------------------------------------------------------------
FEATURES = ["ret", "vol5", "vol21", "vol63", "dd21", "rng21"]


def load_sp500():
    """Real S&P 500 daily close (1990-2022), bundled with the skfolio package."""
    from skfolio.datasets import load_sp500_index
    s = load_sp500_index()
    s.index = pd.to_datetime(s.index)
    return s["SP500"].astype(float).rename("close")


def load_csv_index(path, date_col="Date", close_col="Close"):
    """Load any index (e.g. KSE-100) from a CSV with a date and a close column."""
    d = pd.read_csv(path)
    d[date_col] = pd.to_datetime(d[date_col])
    d[close_col] = pd.to_numeric(d[close_col].astype(str).str.replace(",", ""), errors="coerce")
    d = d.dropna(subset=[close_col]).sort_values(date_col).drop_duplicates(date_col)
    return d.set_index(date_col)[close_col].rename("close")


def make_features(close):
    """Six features from the close price (volume is not available in this data)."""
    r = np.log(close).diff() * 100.0                      # log return in percent
    f = pd.DataFrame(index=close.index)
    f["ret"] = r
    f["vol5"] = r.rolling(5).std()
    f["vol21"] = r.rolling(21).std()
    f["vol63"] = r.rolling(63).std()
    f["dd21"] = np.sqrt((np.minimum(r, 0.0) ** 2).rolling(21).mean())
    f["rng21"] = (close.rolling(21).max() - close.rolling(21).min()) / close.rolling(21).mean() * 100
    return f.dropna()


def scale_on_setup(F, n_setup):
    """z-score with mean/std from the setup period only (frozen)."""
    mu = F.iloc[:n_setup].mean()
    sd = F.iloc[:n_setup].std()
    return (F - mu) / sd, mu, sd


# ----------------------------------------------------------------------------
# 2. Alarm helpers
# ----------------------------------------------------------------------------
def episodes(flags, gap=21, start=0):
    """Turn a daily 0/1 alarm flag into alarm dates: first day of each episode,
    where flags separated by less than `gap` days belong to the same episode."""
    idx = np.flatnonzero(np.asarray(flags)[start:] > 0) + start
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(int(i))
        last = i
    return out


def block_len(x, lo=5, hi=40):
    """Block length for the moving-block bootstrap from the autocorrelation of x^2."""
    z = np.asarray(x, float) ** 2
    z = z - z.mean()
    n = len(z)
    acf = np.correlate(z, z, "full")[n - 1:] / (z @ z)
    lim = 1.96 / np.sqrt(n)
    k = next((i for i in range(1, min(hi, n // 4)) if abs(acf[i]) < lim), hi)
    return int(min(max(k, lo), hi))


def block_len_features(X, cap=63):
    """Block length from the autocorrelation of the FEATURES themselves (largest first-insignificant lag
    over columns, capped at `cap`). Needed because rolling features stay correlated for up to ~60 days."""
    out = []
    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean()
        m = len(x)
        a = np.correlate(x, x, "full")[m - 1:] / (x @ x)
        lim = 1.96 / np.sqrt(m)
        out.append(next((i for i in range(1, min(cap, m // 4)) if a[i] < lim), cap))
    return int(max(3, max(out)))


def mbb_index(n, length, b, rng):
    """Moving-block bootstrap indices (length `length`) from a series of size n."""
    nb = int(np.ceil(length / b))
    st = rng.integers(0, n - b + 1, nb)
    return (st[:, None] + np.arange(b)[None, :]).ravel()[:length]


# ----------------------------------------------------------------------------
# 3. Monitor 5A: covariate drift (window KS + MMD, block-bootstrap calibrated)
# ----------------------------------------------------------------------------
def ks_stat(a, b):
    """Two-sample KS statistic per column; a (n,d), b (m,d)."""
    out = np.empty(a.shape[1])
    for j in range(a.shape[1]):
        sa, sb = np.sort(a[:, j]), np.sort(b[:, j])
        allv = np.concatenate([sa, sb])
        out[j] = np.max(np.abs(np.searchsorted(sa, allv, "right") / len(sa)
                               - np.searchsorted(sb, allv, "right") / len(sb)))
    return out


def _rbf(a, b, gamma):
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d2, 0))


def mmd2(a, b, gamma):
    return _rbf(a, a, gamma).mean() + _rbf(b, b, gamma).mean() - 2 * _rbf(a, b, gamma).mean()


def window_stats(X, t, R, W, gamma, cols=None):
    """KS per feature + MMD between reference X[t-W-R:t-W] and test X[t-W:t]."""
    ref, tst = X[t - W - R:t - W], X[t - W:t]
    return np.append(ks_stat(ref, tst), mmd2(ref, tst, gamma))


def cov_null(Xsetup, R, W, B, gamma, rng, b=None):
    """Block-bootstrap null distribution of the window statistics on the setup period."""
    b = b or block_len_features(Xsetup, W)
    n = len(Xsetup)
    out = np.empty((B, Xsetup.shape[1] + 1))
    for i in range(B):
        Y = Xsetup[mbb_index(n, R + W, b, rng)]
        out[i] = np.append(ks_stat(Y[:R], Y[R:]), mmd2(Y[:R], Y[R:], gamma))
    return out


def cov_monitor(X, n_setup, R=252, W=63, B=1000, stride=1, seed=RNG_SEED):
    """Returns dict with days (indices evaluated), stats (n_eval, d+1) and
    bootstrap p-values (n_eval, d+1). Column d is MMD."""
    rng = np.random.default_rng(seed)
    Xs = X[:n_setup]
    med = np.median(np.sqrt(((Xs[:200, None, :] - Xs[None, :200, :]) ** 2).sum(-1)))
    gamma = 1.0 / (2 * max(med, 1e-6) ** 2)               # median heuristic, fitted on setup
    b = block_len_features(Xs, W)
    null = cov_null(Xs, R, W, B, gamma, rng, b)
    null_sorted = np.sort(null, axis=0)
    days = np.arange(max(n_setup, R + W), len(X), stride)
    stats = np.array([window_stats(X, t, R, W, gamma) for t in days])
    P = np.empty_like(stats)
    for j in range(stats.shape[1]):
        ge = B - np.searchsorted(null_sorted[:, j], stats[:, j], "left")
        P[:, j] = (1 + ge) / (B + 1)
    return dict(days=days, stats=stats, p=P, gamma=gamma, R=R, W=W, B=B, block=b)


def holm_reject(P, alpha, use=None):
    """Holm step-down on rows of P (n, m). `use` selects columns. Returns bool (n,)."""
    if use is not None:
        P = P[:, use]
    n, m = P.shape
    order = np.argsort(P, axis=1)
    Ps = np.take_along_axis(P, order, axis=1)
    thr = alpha / (m - np.arange(m))
    ok = Ps <= thr[None, :]
    # Holm: reject first k hypotheses as long as all previous are rejected; any reject <=> first one rejected
    return ok[:, 0]


def cov_alarm_days(res, alpha=0.01, use=None, n_total=None):
    flags = holm_reject(res["p"], alpha, use)
    full = np.zeros(n_total, int)
    full[res["days"][flags]] = 1
    return full


# ----------------------------------------------------------------------------
# 4. Forecaster (ridge on lagged features, frozen after the setup period)
# ----------------------------------------------------------------------------
def ridge_fit(Xl, y, lam=1.0):
    A = np.c_[np.ones(len(Xl)), Xl]
    I = np.eye(A.shape[1]); I[0, 0] = 0
    return np.linalg.solve(A.T @ A + lam * I, A.T @ y)


def ridge_pred(w, Xl):
    return np.c_[np.ones(len(Xl)), Xl] @ w


def forecast_errors(X, y, n_setup, lam=1.0):
    """e_t = |y_t - yhat_t| where yhat_t uses features at t-1 and a ridge fitted on the setup period.
    Returns error array aligned to y (NaN for t < 1)."""
    Xl = X[:-1]; yt = y[1:]
    w = ridge_fit(Xl[:n_setup - 1], yt[:n_setup - 1], lam)
    e = np.full(len(y), np.nan)
    e[1:] = np.abs(yt - ridge_pred(w, Xl))
    return e, w


# ----------------------------------------------------------------------------
# 5. Streaming detectors
# ----------------------------------------------------------------------------
def run_adwin(x, delta):
    d = ADWIN(delta=delta)
    flags = np.zeros(len(x), int)
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        d.update(float(v))
        if d.drift_detected:
            flags[i] = 1
    return flags


def run_ph(x, lam, delta=0.005):
    d = PageHinkley(min_instances=30, delta=delta, threshold=lam, alpha=0.9999)
    flags = np.zeros(len(x), int)
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        d.update(float(v))
        if d.drift_detected:
            flags[i] = 1
            d = PageHinkley(min_instances=30, delta=delta, threshold=lam, alpha=0.9999)
    return flags


def bocpd_pcp(x, scale0, H=1 / 250., rmax=400, short=10):
    """Bayesian online changepoint detection (Adams & MacKay 2007), Gaussian data with unknown
    mean and variance (Normal-Inverse-Gamma prior). Returns P(run length <= short | x_1..t)."""
    n = len(x)
    mu = np.array([0.0]); ka = np.array([1.0]); al = np.array([1.0]); be = np.array([scale0 ** 2])
    rp = np.array([1.0])
    out = np.zeros(n)
    for t in range(n):
        v = x[t]
        if np.isnan(v):
            out[t] = out[t - 1] if t else 0
            continue
        df = 2 * al
        sc2 = be * (ka + 1) / (al * ka)
        lp = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi * sc2)
              - (df + 1) / 2 * np.log1p((v - mu) ** 2 / (df * sc2)))
        pred = np.exp(lp)
        grow = rp * pred * (1 - H)
        cp = (rp * pred * H).sum()
        rp = np.append(cp, grow)
        s = rp.sum()
        rp = rp / s if s > 0 else np.full(len(rp), 1 / len(rp))
        mu_n = (ka * mu + v) / (ka + 1)
        be_n = be + ka * (v - mu) ** 2 / (2 * (ka + 1))
        mu = np.append(0.0, mu_n); ka = np.append(1.0, ka + 1)
        al = np.append(1.0, al + 0.5); be = np.append(scale0 ** 2, be_n)
        if len(rp) > rmax:
            rp[rmax - 1] += rp[rmax:].sum()
            rp, mu, ka, al, be = rp[:rmax], mu[:rmax], ka[:rmax], al[:rmax], be[:rmax]
        out[t] = rp[:short + 1].sum()
    return out


def cross_up(p, thr):
    """Daily flag when p crosses above thr (from below)."""
    f = np.zeros(len(p), int)
    f[1:] = ((p[1:] > thr) & (p[:-1] <= thr)).astype(int)
    return f


# ----------------------------------------------------------------------------
# 6. Baselines: online HMM, simplified jump model, volatility rule, fixed schedule
# ----------------------------------------------------------------------------
def hmm_flags(Z, n_setup, refit=63, maxfit=756, filt=126, hi=0.8, lo=0.2, seed=RNG_SEED):
    """Online 2-state Gaussian HMM. Refit every `refit` days on past data only; the state probability
    at day t is the filtered probability from the last `filt` days. Alarm on every state switch
    (hysteresis hi/lo). The high-volatility state is the one with the larger mean of column 1."""
    from hmmlearn.hmm import GaussianHMM
    n = len(Z)
    flags = np.zeros(n, int)
    model, hs = None, None
    state = 0
    for t in range(n_setup, n):
        if model is None or (t - n_setup) % refit == 0:
            tr = Z[max(0, t - maxfit):t]
            best = None
            for s in range(2):
                try:
                    m = GaussianHMM(2, covariance_type="diag", n_iter=40, random_state=seed + s)
                    m.fit(tr)
                    ll = m.score(tr)
                    if best is None or ll > best[0]:
                        best = (ll, m)
                except Exception:
                    pass
            if best is not None:
                model = best[1]
                hs = int(np.argmax(model.means_[:, 1]))
        if model is None:
            continue
        p = model.predict_proba(Z[max(0, t - filt + 1):t + 1])[-1, hs]
        new = 1 if p > hi else (0 if p < lo else state)
        if new != state:
            flags[t] = 1
            state = new
    return flags


def jump_flags(Z, n_setup, lam=8.0, refit=63, maxfit=756, iters=10, seed=RNG_SEED):
    """Simplified (non-sparse) 2-state statistical jump model. Centres are fitted by alternating
    Viterbi-with-jump-penalty and mean updates on past data; the online state is the arg-min of a
    forward cost recursion. Alarm on every state switch."""
    n = len(Z)
    flags = np.zeros(n, int)
    cen, V, state = None, None, 0
    rng = np.random.default_rng(seed)
    for t in range(n_setup, n):
        if cen is None or (t - n_setup) % refit == 0:
            tr = Z[max(0, t - maxfit):t]
            c = tr[rng.choice(len(tr), 2, replace=False)]
            c = c[np.argsort(c[:, 0])]
            for _ in range(iters):
                d = ((tr[:, None, :] - c[None]) ** 2).sum(-1)
                cost = d[0].copy(); bp = np.zeros((len(tr), 2), int)
                for i in range(1, len(tr)):
                    stay, sw = cost, cost[::-1] + lam
                    bp[i] = np.where(sw < stay, 1 - np.arange(2), np.arange(2))
                    cost = np.minimum(stay, sw) + d[i]
                s = np.empty(len(tr), int); s[-1] = int(np.argmin(cost))
                for i in range(len(tr) - 1, 0, -1):
                    s[i - 1] = bp[i][s[i]]
                for k in range(2):
                    if (s == k).sum() > 5:
                        c[k] = tr[s == k].mean(0)
            cen = c[np.argsort(c[:, 0])]
            V = np.zeros(2)
        d = ((Z[t] - cen) ** 2).sum(-1)
        V = np.minimum(V, V[::-1] + lam) + d
        V -= V.min()
        new = int(np.argmin(V))
        if new != state:
            flags[t] = 1
            state = new
    return flags


def vol_rule_flags(vol21, n_setup, mult=2.0):
    """Alarm when 21-day volatility crosses above mult x its setup-period median."""
    thr = mult * np.median(vol21[:n_setup])
    return cross_up(np.asarray(vol21, float), thr)


def schedule_flags(n, n_setup, every=252):
    f = np.zeros(n, int)
    f[n_setup + every - 1::every] = 1
    return f


# ----------------------------------------------------------------------------
# 7. Calibration on the setup period (block bootstrap of the setup series)
# ----------------------------------------------------------------------------
def calibrate_stream(x_setup, kind, grid, rng, B=60, years=4.0, target=1.0, scale0=1.0):
    """Choose the MOST SENSITIVE setting whose bootstrap false-alarm rate on the (alarm-free)
    setup series is <= target alarms/year. `grid` is ordered from least to most sensitive."""
    n = len(x_setup)
    b = block_len(x_setup)
    L = int(252 * years)
    if kind == "bocpd":
        pcps = [bocpd_pcp(x_setup[mbb_index(n, L, b, rng)], scale0)[40:] for _ in range(B // 3)]
    far = {}
    for g in grid:
        counts = []
        for i in range(B // 3 if kind == "bocpd" else B):
            if kind == "bocpd":
                fl = cross_up(pcps[i], g)
            else:
                xs = x_setup[mbb_index(n, L, b, rng)]
                fl = run_adwin(xs, g) if kind == "adwin" else run_ph(xs, g)
            counts.append(len(episodes(fl)) / (len(fl) / 252.0))
        far[g] = float(np.mean(counts))
    ok = [g for g in grid if far[g] <= target]
    return (ok[-1] if ok else grid[0]), far


ADWIN_GRID = [1e-5, 1e-4, 1e-3, 5e-3, 2e-2, 5e-2, 1e-1, 2e-1, 3e-1]            # increasing delta = more sensitive
PH_GRID = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]                   # decreasing threshold = more sensitive
BOCPD_GRID = [0.95, 0.9, 0.8, 0.6, 0.4, 0.2]                       # decreasing = more sensitive


# ----------------------------------------------------------------------------
# 8. Evaluation against dated events (real data)
# ----------------------------------------------------------------------------
def evaluate_events(alarms, events, n, start, tol=10):
    """alarms, events: lists of integer day indices. Returns dict of metrics.
    TP alarm = within +-tol days of an event. Delay = first alarm in the window minus event day
    (negative = lead). FAR = false alarms per year over [start, n)."""
    alarms = [a for a in alarms if a >= start]
    tp, delays = 0, []
    for a in alarms:
        if any(abs(a - e) <= tol for e in events):
            tp += 1
    for e in events:
        w = [a for a in alarms if abs(a - e) <= tol]
        delays.append(min(w) - e if w else np.nan)
    years = (n - start) / 252.0
    det = [d for d in delays if not np.isnan(d)]
    fp = len(alarms) - tp
    return dict(n_alarms=len(alarms), precision=tp / len(alarms) if alarms else np.nan,
                recall=len(det) / len(events), far=fp / years,
                median_delay=float(np.median(det)) if det else np.nan,
                lead=float(np.mean([max(0, -d) for d in det])) if det else np.nan,
                delays=delays)


# ----------------------------------------------------------------------------
# 9. Synthetic generator with known drift dates
# ----------------------------------------------------------------------------
BETA = np.array([0.6, -0.4, 0.3])
BETA2 = np.array([0.3, 0.6, -0.4])          # different direction, same-ish norm (concept-only 'rotate')
SPECS = {
    "cov_mean_0.5": dict(mean=0.5),
    "cov_mean_1": dict(mean=1.0),
    "cov_mean_2": dict(mean=2.0),
    "cov_var_2": dict(var=2.0),
    "cov_var_3": dict(var=3.0),
    "concept_flip": dict(beta=-BETA),
    "concept_rot": dict(beta=BETA2),
    "joint": dict(mean=1.0, beta=-BETA),
}
KIND = {k: ("covariate" if k.startswith("cov") else "concept" if k.startswith("concept") else "joint")
        for k in SPECS}


def simulate(spec, T=1800, tau=1100, seed=0):
    """3 AR(1) features with t(5) shocks and stochastic volatility; y_t = beta' x_{t-1} + noise.
    `spec` None gives a no-drift (null) series. Returns X (T,3), y (T,), tau."""
    rng = np.random.default_rng(seed)
    d = 3
    ls = np.zeros(T)
    for t in range(1, T):
        ls[t] = 0.98 * ls[t - 1] + 0.08 * rng.standard_normal()
    sig = np.exp(ls)
    sh = rng.standard_t(5, (T, d)) / np.sqrt(5 / 3.0)
    x0 = np.zeros((T, d))
    for t in range(1, T):
        x0[t] = 0.5 * x0[t - 1] + np.sqrt(1 - 0.25) * sig[t] * sh[t]
    noise = rng.standard_t(5, T) / np.sqrt(5 / 3.0) * sig
    X = x0.copy()
    beta_t = np.tile(BETA, (T, 1))
    if spec:
        s = SPECS[spec]
        if "var" in s:
            X[tau:] *= np.sqrt(s["var"])
        if "mean" in s:
            X[tau:] += s["mean"]
        if "beta" in s:
            beta_t[tau:] = s["beta"]
    y = np.zeros(T)
    y[1:] = (beta_t[1:] * X[:-1]).sum(1) + 1.0 * noise[1:]
    return X, y, (tau if spec else None)


def chance_reference(n_alarms, events, n, start, tol=10, reps=4000, seed=1):
    """Random alarms (same count, uniform over the test period): expected recall/precision and
    the probability of reaching `observed` recall by luck is computed by the caller."""
    rng = np.random.default_rng(seed)
    rec = np.empty(reps); prec = np.empty(reps)
    for i in range(reps):
        a = np.sort(rng.integers(start, n, n_alarms))
        hit = [any(abs(x - e) <= tol for x in a) for e in events]
        tp = sum(any(abs(x - e) <= tol for e in events) for x in a)
        rec[i] = np.mean(hit); prec[i] = tp / max(n_alarms, 1)
    return rec, prec
