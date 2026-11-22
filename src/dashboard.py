"""dashboard.py - interactive drift dashboard (plotly). Full version + anonymised study stimuli."""
import os, pickle, numpy as np, pandas as pd
import paths as P
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import drift as D

R = pickle.load(open(P.RESULTS/"results_SP500.pkl", "rb"))
dates, n, cov, fam = R["dates"], R["n"], R["cov"], R["fam"]
close = R["close"].values


def build(lo=0, hi=None, anonymize=False, plotlyjs=True, title="Drift dashboard"):
    hi = hi or n
    sl = slice(lo, hi)
    x = np.arange(lo, hi) - lo if anonymize else dates[sl]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.3, 0.3, 0.17, 0.23],
                        subplot_titles=("Price (rebased) and alarms", "Feature distribution shift (KS statistic, 63 d vs 252 d)",
                                        "Detector agreement (monitors in alarm, of 3)", "Uncertainty / evidence"))
    px = close[sl] / close[lo] * 100
    fig.add_trace(go.Scatter(x=x, y=px, name="Price", line=dict(color="#1f4e79", width=1.4)), 1, 1)
    for key, col, nm, off in (("cov", "#6c3483", "KS/MMD alarm (features)", 0.0), ("err", "#2e8b57", "ADWIN/PH alarm (error)", 0.04),
                              ("bocpd", "#e08e0b", "BOCPD alarm", 0.08)):
        a = [i for i in fam[key] if lo <= i < hi]
        fig.add_trace(go.Scatter(x=[(i - lo) if anonymize else dates[i] for i in a], y=[px.min() * (0.97 - off)] * len(a),
                                 mode="markers", marker=dict(symbol="triangle-up", size=10, color=col), name=nm,
                                 hovertemplate="alarm day %{x}<extra>" + nm + "</extra>"), 1, 1)
    days = cov["days"]; m = (days >= lo) & (days < hi)
    fig.add_trace(go.Heatmap(x=([d - lo for d in days[m]] if anonymize else dates[days[m]]), y=D.FEATURES,
                             z=cov["stats"][m][:, :6].T, colorscale="Magma", zmin=0, zmax=1, showscale=False,
                             hovertemplate="%{y}: KS=%{z:.2f}<extra></extra>"), 2, 1)
    agree = np.zeros(n)
    for k in fam:
        f = np.zeros(n)
        for a in fam[k]:
            f[a:a + 11] = 1
        agree += f
    fig.add_trace(go.Scatter(x=x, y=agree[sl], fill="tozeroy", line=dict(color="#c0392b", shape="hv"), name="monitors in alarm"), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=R["bocpd_pcp"][sl], name="BOCPD: P(change in last 10 d)", line=dict(color="#e08e0b", width=1)), 4, 1)
    ev = np.full(n, np.nan); ev[days] = np.clip(-np.log10(cov["p"].min(1)) / 3, 0, 1)
    fig.add_trace(go.Scatter(x=x, y=ev[sl], name="KS/MMD evidence (-log10 p / 3)", line=dict(color="#6c3483", width=1)), 4, 1)
    if not anonymize:
        for k, i in R["events"].items():
            if lo <= i < hi:
                fig.add_vline(x=dates[i], line=dict(color="#c0392b", dash="dash", width=1), annotation_text=k.split(" ")[0] + " " + k.split(" ")[1],
                              annotation_font_size=9)
    fig.update_layout(height=900, title=title, template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.08), margin=dict(l=60, r=20, t=80, b=40))
    if not anonymize:
        fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.04, row=4, col=1)
    else:
        fig.update_xaxes(title_text="trading day in window", row=4, col=1)
    return fig.to_html(include_plotlyjs=True if plotlyjs else "cdn", full_html=True)


if __name__ == "__main__":
    open(P.OUT/"dashboard.html", "w", encoding="utf-8").write(build(title="Watching the Distribution Move: S&P 500 drift dashboard"))
    print("ok", os.path.getsize(P.OUT/"dashboard.html") // 1000, "KB")
